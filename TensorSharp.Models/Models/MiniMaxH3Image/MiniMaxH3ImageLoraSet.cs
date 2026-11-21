// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image LoRA plug-ins (docs/design/diffusion-runtime.md section 5.3): any number of style adapters, each with its
// own strength, and at most one speed adapter, loaded from any of the three key grammars (diffusers PEFT, ComfyUI native,
// kohya), validated tensor by tensor and packed per projection for the denoiser's LoraLinear regions, applied unmerged:
// y = W x + up (down x) with the base weights as stored.
//
// Where the terms land (MiniMaxH3LoraKeys): diffusers' to_q / to_k / to_v on the stacked thirds of attn.qkv_proj, its
// ff.net.0.proj with lora_B's two halves swapped from [value; gate] into fc1's [gate | value]; a ComfyUI or kohya qkv_proj
// term over all three thirds as stored (a block-diagonal B, Comfy's conversion of split q/k/v, splits into its three
// thirds exactly: the columns of a third that are all zero carry nothing to it), and fc1 as stored.
//
// strength * alpha / rank is folded into up in F32. Alpha: diffusers from the file's metadata ("alpha", "lora_alpha", or
// the PEFT config in "lora_adapter_metadata"), else the plug-in config's, else the rank; ComfyUI and kohya from each
// module's .alpha tensor, else the config's, else the rank (ComfyUI's rule: no alpha is scale 1). A config alpha the file
// contradicts is refused.
//
// Several adapters on one projection are concatenated along the rank (Qwen-Image-2.1's packing), so a projection takes one
// shrink and one expand per slot whatever the adapter count. Each rank component is balanced (down row and up column of
// equal norm; the product is unchanged) and stored F16 unless a value would reach F16's range (then F32) or the backend is
// ggml_cpu (F32 always: its ARM kernels accumulate F16 dot products in F16) or the managed executor (F32). Ranks are
// zero-padded to 64 on Metal (its simdgroup kernel needs K >= 64) and to 16 elsewhere; the managed executor keeps them.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.SharedDiffusion;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>The loaded LoRA plug-ins of a MiniMax-H3-Image denoiser, packed per projection.</summary>
    internal sealed class MiniMaxH3ImageLoraSet : IDisposable
    {
        /// <summary>The block projections a term can update.</summary>
        internal enum Projection { Qkv = 0, Out = 1, Fc1 = 2, Fc2 = 3 }

        internal const int ProjectionCount = 4;
        private const int MaxPackingThreads = 8;
        private static long s_generations;

        /// <summary>Test seam: when set, <see cref="InvalidateDeviceCopies"/> hands each factor pointer to it instead of
        /// <see cref="GgmlBasicOps.InvalidateHostBuffer"/>.</summary>
        internal static Action<IntPtr> InvalidateHookForTest;

        /// <summary>One packed factor pair: a slot's rows of a projection, every adapter's terms concatenated along the rank.</summary>
        internal sealed class Part
        {
            /// <summary>The slot (rows [Slot * slot rows, (Slot + 1) * slot rows) of the projection's output).</summary>
            public int Slot;
            /// <summary><see cref="GraphFormat.F16"/> or <see cref="GraphFormat.F32"/>, both factors.</summary>
            public int Format;
            /// <summary>The (padded) rank; the padding is zero.</summary>
            public int Rank;
            public long In, Out;
            /// <summary>down: ggml [ne0 = in, ne1 = rank]; up: ggml [ne0 = rank, ne1 = out], the scales folded in.</summary>
            public IntPtr Down, Up;
            public long DownBytes, UpBytes;
        }

        /// <summary>A projection's parts: <see cref="SlotRows"/> 0 is one slot of all its rows; else equal slots (q, k, v).</summary>
        internal sealed class Factors
        {
            public long SlotRows;
            public Part[] Parts;
        }

        /// <summary>One plug-in as loaded, for reports and the runtime's <see cref="AdapterSet"/>.</summary>
        internal sealed class Adapter
        {
            public string Name, WeightsFile, ConfigFile;
            public MiniMaxH3LoraGrammar Grammar;
            public float Strength;
            public SamplingContract Contract;
            public int Tensors, Terms, TrunkBlocks, RefinerBlocks;
            public IReadOnlyList<int> Ranks = Array.Empty<int>();
            public IReadOnlyList<float> Scales = Array.Empty<float>();
            public string AlphaSource;
        }

        /// <summary>A planned term: one module's factors in one file.</summary>
        private sealed class Term
        {
            public int Adapter;
            public SafetensorsFile File;
            public string FileName;
            public bool Refiner;
            public int Block;
            public MiniMaxH3LoraTarget Target;
            public string DownName, UpName;
            public int Rank;
            public long In, Out;
            public float Scale;
            public bool Swap;   // diffusers' ff.net.0.proj: [value; gate] -> [gate | value]
        }

        /// <summary>A conditioned term's piece of one slot: down [Rank, In], up [Out, Rank] row-major F32, scale folded.</summary>
        private sealed record Piece(float[] Down, float[] Up, int Rank);

        private readonly LoraFactorBuffers _buffers = new();
        private readonly object _lock = new();
        private readonly MiniMaxH3Config _cfg;
        private readonly Factors[] _trunk, _refiner;
        private readonly List<Adapter> _adapters = new();
        private long _factorBytes;
        private int _f16Parts, _f32Parts;
        private bool _disposed;

        private MiniMaxH3ImageLoraSet(MiniMaxH3Config cfg, BackendType backend)
        {
            _cfg = cfg;
            Backend = backend;
            _trunk = new Factors[cfg.NumLayers * ProjectionCount];
            _refiner = new Factors[cfg.TokenRefinerLayers * ProjectionCount];
            Generation = Interlocked.Increment(ref s_generations);
        }

        /// <summary>A process-unique id: a denoiser's cached plan is bound to one set, never to an address a later set reuses.</summary>
        internal long Generation { get; }

        /// <summary>The backend the factors are packed for (their element type and rank padding).</summary>
        internal BackendType Backend { get; }

        /// <summary>The checkpoint's shape the set was validated against.</summary>
        internal MiniMaxH3Config Config => _cfg;

        internal IReadOnlyList<Adapter> Adapters => _adapters;

        /// <summary>The runtime's view: one <see cref="DiffusionAdapter"/> per plug-in, in order, each carrying this set.</summary>
        internal AdapterSet AdapterSet { get; private set; } = AdapterSet.Empty;

        /// <summary>The speed adapter's contract, or null.</summary>
        internal SamplingContract Contract => AdapterSet.Contract;

        /// <summary>Host bytes of the packed factors (the device takes as many once they are bound).</summary>
        internal long FactorBytes => Interlocked.Read(ref _factorBytes);

        /// <summary>The load report, one line a plug-in.</summary>
        internal string Summary { get; private set; } = "";

        /// <summary>The packed factors of a projection, or null when no adapter updates it.</summary>
        internal Factors Get(bool refiner, int block, Projection projection)
        {
            int blocks = refiner ? _cfg.TokenRefinerLayers : _cfg.NumLayers;
            if (block < 0 || block >= blocks)
                throw new ArgumentOutOfRangeException(nameof(block), $"{(refiner ? "Refiner block" : "Block")} {block} is outside [0, {blocks}).");
            return (refiner ? _refiner : _trunk)[block * ProjectionCount + (int)projection];
        }

        /// <summary>Whether the set was loaded for a checkpoint of <paramref name="cfg"/>'s shape.</summary>
        internal bool Fits(MiniMaxH3Config cfg) =>
            cfg != null && cfg.NumLayers == _cfg.NumLayers && cfg.TokenRefinerLayers == _cfg.TokenRefinerLayers &&
            cfg.HiddenSize == _cfg.HiddenSize && cfg.AttentionInnerDim == _cfg.AttentionInnerDim && cfg.FfnHiddenSize == _cfg.FfnHiddenSize;

        // ---- loading --------------------------------------------------------------------------------------------------

        /// <summary>
        /// Load <paramref name="resolved"/> (specs after <see cref="LoraCliFlags.Resolve"/>: the weights, with a plug-in config
        /// as <see cref="LoraSpec.ConfigPath"/> or none for a bare style adapter) for a denoiser of <paramref name="cfg"/> on
        /// <paramref name="backend"/>.
        /// </summary>
        /// <exception cref="ArgumentException">A spec is refused (another family's config, two speed adapters, weights the config
        /// was not written for, a negative strength), with the canonical text.</exception>
        /// <exception cref="InvalidDataException">A config or a weights file is malformed or does not fit the transformer; every
        /// problem of a file is named in one message.</exception>
        /// <exception cref="FileNotFoundException">A named file is missing.</exception>
        internal static MiniMaxH3ImageLoraSet Load(IReadOnlyList<LoraSpec> resolved, MiniMaxH3Config cfg, BackendType backend)
        {
            ArgumentNullException.ThrowIfNull(resolved);
            ArgumentNullException.ThrowIfNull(cfg);
            if (resolved.Count == 0) throw new ArgumentException("MiniMax-H3-Image LoRA: no plug-in to load.", nameof(resolved));
            var set = new MiniMaxH3ImageLoraSet(cfg, backend);
            var files = new List<SafetensorsFile>();
            try
            {
                var terms = new List<Term>();
                for (int i = 0; i < resolved.Count; i++)
                    terms.AddRange(set.Plan(i, resolved[i], files));
                set.AdapterSet = new AdapterSet(set._adapters.Select(a => new DiffusionAdapter
                {
                    Name = a.Name, Scale = a.Strength, Contract = a.Contract, Payload = set,
                }));
                set.Pack(terms, files);
                set.Summary = string.Join(Environment.NewLine, set._adapters.Select(set.Describe));
                return set;
            }
            catch
            {
                set.Dispose();
                throw;
            }
            finally
            {
                foreach (SafetensorsFile f in files) f.Dispose();
            }
        }

        /// <summary>The spec's config (null for bare weights) and its weights, with the checks of
        /// <see cref="MiniMaxH3ImageLoraConfig.ReadConfigs"/>; then every tensor of the file grouped into terms.</summary>
        private List<Term> Plan(int index, LoraSpec spec, List<SafetensorsFile> files)
        {
            ArgumentNullException.ThrowIfNull(spec);
            string manifest = MiniMaxH3ImageLoraConfig.ManifestOf(spec);
            MiniMaxH3ImageLoraConfig config = null;
            string weights = spec.Path;
            if (manifest != null)
            {
                string declared = LoraCliFlags.DeclaredPluginType(new LoraSpec(manifest));
                if (declared != MiniMaxH3ImageLoraConfig.Type)
                    throw new ArgumentException(MiniMaxH3ImageLoraConfig.ForeignConfigRefusal(Path.GetFileName(spec.Path), declared));
                config = MiniMaxH3ImageLoraConfig.Load(manifest);
                if (string.Equals(Path.GetFullPath(spec.Path), config.Path, StringComparison.Ordinal))
                {
                    // A spec that still names the config itself uses its weights entry, resolved without a download.
                    if (spec.ConfigPath != null)
                        throw new ArgumentException(
                            $"{LoraCliFlags.LoraFlag} {config.Path} is a plug-in config already; drop the {LoraCliFlags.ConfigFlag} after it.");
                    weights = ConfigFileArgs.ResolveFileEntryPath(config.Path, "weights");
                    if (weights == null || !File.Exists(weights))
                        throw new FileNotFoundException($"{LoraCliFlags.LoraFlag} file not found: {weights}", weights);
                }
                config.CheckWeights(weights);
            }
            else if (!File.Exists(weights))
                throw new FileNotFoundException($"{LoraCliFlags.LoraFlag} file not found: {weights}", weights);
            if (!weights.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException($"{LoraCliFlags.LoraFlag} {Path.GetFileName(weights)}: MiniMax-H3-Image LoRA weights are a .safetensors file.");

            string fileName = Path.GetFileName(weights);
            string name = config?.Name ?? Path.GetFileNameWithoutExtension(weights);
            float strength = spec.Scale ?? config?.Scale ?? 1f;
            MiniMaxH3ImageLoraConfig.CheckStrength(name, fileName, strength, config?.Contract != null);

            var st = new SafetensorsFile(Path.GetFullPath(weights));
            files.Add(st);
            var adapter = new Adapter
            {
                Name = name,
                WeightsFile = fileName,
                ConfigFile = config != null ? Path.GetFileName(config.Path) : null,
                Strength = strength,
                Contract = config?.Contract,
            };
            List<Term> terms = PlanFile(index, st, fileName, config, strength, adapter);
            _adapters.Add(adapter);
            return terms;
        }

        private List<Term> PlanFile(int index, SafetensorsFile st, string fileName, MiniMaxH3ImageLoraConfig config, float strength, Adapter adapter)
        {
            var names = new List<string>(st.Tensors.Keys);
            names.Sort(StringComparer.Ordinal);
            adapter.Tensors = names.Count;
            var rejected = new List<string>();
            var slots = new SortedSet<string>(StringComparer.Ordinal);
            var grammars = new SortedDictionary<MiniMaxH3LoraGrammar, string>();
            var parts = new Dictionary<(bool Refiner, int Block, MiniMaxH3LoraTarget Target), (string A, string B, string Alpha)>();
            foreach (string name in names)
            {
                // A name in a grammar the file does not otherwise use is a mix even when it does not resolve.
                if (MiniMaxH3LoraKeys.GrammarOf(name) is { } g && !grammars.ContainsKey(g)) grammars[g] = name;
                var key = MiniMaxH3LoraKeys.ResolveAny(name, _cfg.NumLayers, _cfg.TokenRefinerLayers, out string reason);
                if (key == null)
                {
                    rejected.Add($"{name}: {reason}");
                    continue;
                }
                var k = key.Value;
                if (k.Grammar == MiniMaxH3LoraGrammar.Diffusers) slots.Add(k.Slot ?? "(none)");
                var id = (k.Refiner, k.Block, k.Target);
                parts.TryGetValue(id, out var pair);
                string existing = k.Role switch { MiniMaxH3LoraRole.Down => pair.A, MiniMaxH3LoraRole.Up => pair.B, _ => pair.Alpha };
                if (existing != null)
                {
                    rejected.Add($"{name}: a second {(k.Role == MiniMaxH3LoraRole.Alpha ? "alpha" : k.Role == MiniMaxH3LoraRole.Down ? "down factor" : "up factor")} " +
                                 $"for {ModuleName(k.Grammar, k.Refiner, k.Block, k.Target)} (beside {existing})");
                    continue;
                }
                parts[id] = k.Role switch
                {
                    MiniMaxH3LoraRole.Down => (name, pair.B, pair.Alpha),
                    MiniMaxH3LoraRole.Up => (pair.A, name, pair.Alpha),
                    _ => (pair.A, pair.B, name),
                };
            }
            if (grammars.Count > 1)
                rejected.Insert(0, "the file mixes key grammars (" +
                    string.Join(", ", grammars.Select(g => $"{MiniMaxH3LoraKeys.Spell(g.Key)}: {g.Value}")) + "); a LoRA file is written in one");
            if (slots.Count > 1)
                rejected.Insert(0, $"several PEFT adapter slots ({string.Join(", ", slots)}); export the one to use on its own");
            MiniMaxH3LoraGrammar grammar = grammars.Count > 0 ? grammars.Keys.First() : MiniMaxH3LoraGrammar.Diffusers;
            adapter.Grammar = grammar;
            if (config != null && config.KeyFormat != MiniMaxH3ImageLoraKeyFormat.Auto && grammars.Count == 1 &&
                !string.Equals(config.KeyFormat.ToString(), grammar.ToString(), StringComparison.Ordinal))
                rejected.Insert(0, $"{Path.GetFileName(config.Path)} declares key_format \"{config.KeyFormat.ToString().ToLowerInvariant()}\", " +
                                   $"but the names are {MiniMaxH3LoraKeys.Spell(grammar)} (e.g. {grammars.Values.First()})");

            // ---- alpha
            float? fileAlpha = null;
            string alphaSource;
            if (grammar == MiniMaxH3LoraGrammar.Diffusers)
            {
                fileAlpha = MetadataAlpha(st, fileName, rejected);
                if (config?.Alpha is { } c && fileAlpha is { } f && c != f)
                    rejected.Insert(0, $"{Path.GetFileName(config.Path)} sets alpha {MiniMaxH3LoraConfig.Num(c)}, but the file's metadata says {MiniMaxH3LoraConfig.Num(f)}");
                alphaSource = fileAlpha != null ? "metadata" : config?.Alpha != null ? "config" : "rank";
            }
            else
            {
                bool tensors = parts.Values.Any(p => p.Alpha != null);
                alphaSource = tensors ? ".alpha tensors" : config?.Alpha != null ? "config" : "rank";
            }

            var terms = new List<Term>();
            var ranks = new SortedSet<int>();
            var scales = new SortedSet<float>();
            var trunkBlocks = new HashSet<int>();
            var refinerBlocks = new HashSet<int>();
            foreach (var entry in parts.OrderBy(p => p.Key.Refiner ? 0 : 1).ThenBy(p => p.Key.Block).ThenBy(p => (int)p.Key.Target))
            {
                var (refiner, block, target) = entry.Key;
                var (a, b, alphaName) = entry.Value;
                string module = ModuleName(grammar, refiner, block, target);
                if (a == null || b == null)
                {
                    rejected.Add(a == null && b == null
                        ? $"{alphaName}: an .alpha without its module's factors"
                        : $"{module}: has only its {(a == null ? "up" : "down")} factor; a term needs both");
                    continue;
                }
                var (input, output) = MiniMaxH3LoraKeys.Shape(_cfg, target);
                SafetensorTensorInfo down = st.GetInfo(a), up = st.GetInfo(b);
                string bad = DtypeProblem(down) ?? DtypeProblem(up);
                if (bad != null)
                {
                    rejected.Add($"{module}: {bad}");
                    continue;
                }
                if (down.Shape.Length != 2 || up.Shape.Length != 2 || down.Shape[0] < 1 || down.Shape[0] > int.MaxValue ||
                    down.Shape[1] != input || up.Shape[0] != output || up.Shape[1] != down.Shape[0])
                {
                    rejected.Add(
                        $"{module}: down [{string.Join(",", down.Shape)}] / up [{string.Join(",", up.Shape)}], but " +
                        $"{MiniMaxH3LoraKeys.GgufTarget(refiner, block, target, _cfg.AttentionInnerDim)} is {input} -> {output} " +
                        "(down [rank, in], up [out, rank]; a LoRA made for a different model)");
                    continue;
                }
                int rank = (int)down.Shape[0];
                float alpha;
                if (grammar == MiniMaxH3LoraGrammar.Diffusers)
                    alpha = fileAlpha ?? config?.Alpha ?? rank;
                else if (alphaName != null)
                {
                    float? t = TensorAlpha(st, alphaName, rejected);
                    if (t == null) continue;
                    if (config?.Alpha is { } c && c != t.Value)
                    {
                        rejected.Add($"{alphaName} is {MiniMaxH3LoraConfig.Num(t.Value)}, but {Path.GetFileName(config.Path)} sets alpha {MiniMaxH3LoraConfig.Num(c)}");
                        continue;
                    }
                    alpha = t.Value;
                }
                else
                    alpha = config?.Alpha ?? rank;
                var term = new Term
                {
                    Adapter = index, File = st, FileName = fileName, Refiner = refiner, Block = block, Target = target,
                    DownName = a, UpName = b, Rank = rank, In = input, Out = output, Scale = strength * alpha / rank,
                    Swap = target == MiniMaxH3LoraTarget.Fc1 && grammar == MiniMaxH3LoraGrammar.Diffusers,
                };
                terms.Add(term);
                ranks.Add(rank);
                scales.Add(term.Scale);
                (refiner ? refinerBlocks : trunkBlocks).Add(block);
            }

            if (rejected.Count > 0)
            {
                const int Shown = 6;
                var shown = rejected.Count <= Shown ? rejected : rejected.GetRange(0, Shown);
                throw new InvalidDataException(
                    $"LoRA '{fileName}' does not fit MiniMax-H3-Image ({rejected.Count} problem(s)): {string.Join("; ", shown)}" +
                    (rejected.Count > Shown ? $"; ... and {rejected.Count - Shown} more" : "") +
                    ". Nothing is skipped silently: a partly applied LoRA would not be the adapter you asked for.");
            }
            if (terms.Count == 0)
                throw new InvalidDataException($"LoRA '{fileName}' contains no tensors.");
            adapter.Terms = terms.Count;
            adapter.TrunkBlocks = trunkBlocks.Count;
            adapter.RefinerBlocks = refinerBlocks.Count;
            adapter.Ranks = ranks.ToArray();
            adapter.Scales = scales.ToArray();
            adapter.AlphaSource = alphaSource;
            return terms;
        }

        /// <summary>A diffusers file's alpha: "alpha" or "lora_alpha" in its metadata, or PEFT's adapter config in
        /// "lora_adapter_metadata"; null when it states none. A PEFT config whose alpha or rank varies by module, or that uses
        /// rsLoRA or DoRA, is refused.</summary>
        private static float? MetadataAlpha(SafetensorsFile st, string fileName, List<string> rejected)
        {
            var found = new List<(string Key, float Value)>();
            foreach (string key in new[] { "alpha", "lora_alpha" })
            {
                if (!st.Metadata.TryGetValue(key, out string raw)) continue;
                if (!float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float a) || !float.IsFinite(a) || !(a > 0f))
                {
                    rejected.Add($"__metadata__ \"{key}\" is '{raw}', not a positive number");
                    continue;
                }
                found.Add((key, a));
            }
            if (st.Metadata.TryGetValue("lora_adapter_metadata", out string peft))
            {
                try
                {
                    using var doc = JsonDocument.Parse(peft);
                    JsonElement r = doc.RootElement;
                    if (r.TryGetProperty("lora_alpha", out var la) && la.ValueKind == JsonValueKind.Number && la.TryGetSingle(out float pa) && pa > 0f)
                        found.Add(("lora_adapter_metadata.lora_alpha", pa));
                    foreach (string pattern in new[] { "alpha_pattern", "rank_pattern" })
                        if (r.TryGetProperty(pattern, out var p) && p.ValueKind == JsonValueKind.Object && p.EnumerateObject().Any())
                            rejected.Add($"__metadata__ lora_adapter_metadata has a per-module {pattern}, which TensorSharp does not apply");
                    foreach (string flag in new[] { "use_rslora", "use_dora" })
                        if (r.TryGetProperty(flag, out var f) && f.ValueKind == JsonValueKind.True)
                            rejected.Add($"__metadata__ lora_adapter_metadata sets {flag}, which TensorSharp does not apply");
                }
                catch (JsonException e)
                {
                    rejected.Add($"__metadata__ lora_adapter_metadata is not JSON: {e.Message}");
                }
            }
            if (found.Count == 0) return null;
            if (found.Any(f => f.Value != found[0].Value))
                rejected.Add($"__metadata__ states different alphas ({string.Join(", ", found.Select(f => $"{f.Key} {MiniMaxH3LoraConfig.Num(f.Value)}"))})");
            return found[0].Value;
        }

        /// <summary>A module's .alpha tensor: one number of any numeric dtype.</summary>
        private static float? TensorAlpha(SafetensorsFile st, string name, List<string> rejected)
        {
            SafetensorTensorInfo info = st.GetInfo(name);
            if (info.NumElements != 1)
            {
                rejected.Add($"{name}: an alpha holds one number, not {info.NumElements}");
                return null;
            }
            byte[] raw = st.ReadRawBytes(name);
            double value = info.Dtype switch
            {
                SafetensorDtype.F32 => BitConverter.ToSingle(raw),
                SafetensorDtype.F64 => BitConverter.ToDouble(raw),
                SafetensorDtype.F16 => (double)BitConverter.ToHalf(raw),
                SafetensorDtype.BF16 => BitConverter.UInt32BitsToSingle((uint)BitConverter.ToUInt16(raw) << 16),
                SafetensorDtype.I64 => BitConverter.ToInt64(raw),
                SafetensorDtype.I32 => BitConverter.ToInt32(raw),
                SafetensorDtype.I16 => BitConverter.ToInt16(raw),
                _ => double.NaN,
            };
            if (!double.IsFinite(value) || !(value > 0) || !float.IsFinite((float)value))
            {
                rejected.Add($"{name}: alpha is {(double.IsNaN(value) ? info.Dtype.ToString() : value.ToString(CultureInfo.InvariantCulture))}, not a positive number");
                return null;
            }
            return (float)value;
        }

        private static string DtypeProblem(SafetensorTensorInfo info) =>
            info.Dtype is SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32
                ? null
                : $"{info.Name} is {info.Dtype}; LoRA factors must be BF16, F16 or F32";

        /// <summary>A module's name in its grammar, for messages.</summary>
        internal static string ModuleName(MiniMaxH3LoraGrammar grammar, bool refiner, int block, MiniMaxH3LoraTarget target)
        {
            string b = block.ToString(CultureInfo.InvariantCulture);
            switch (grammar)
            {
                case MiniMaxH3LoraGrammar.Diffusers:
                    return MiniMaxH3LoraKeys.LoraModule(refiner, block, target);
                case MiniMaxH3LoraGrammar.Comfy:
                    return $"diffusion_model.{(refiner ? "token_refiner.blocks" : "blocks")}.{b}." + target switch
                    {
                        MiniMaxH3LoraTarget.Out => "attn.out_proj",
                        MiniMaxH3LoraTarget.Fc1 => "mlp.fc1",
                        MiniMaxH3LoraTarget.Fc2 => "mlp.fc2",
                        _ => "attn.qkv_proj",
                    };
                default:
                    return $"lora_unet_{(refiner ? "token_refiner_blocks" : "blocks")}_{b}_" + target switch
                    {
                        MiniMaxH3LoraTarget.Out => "attn_out_proj",
                        MiniMaxH3LoraTarget.Fc1 => "mlp_fc1",
                        MiniMaxH3LoraTarget.Fc2 => "mlp_fc2",
                        _ => "attn_qkv_proj",
                    };
            }
        }

        // ---- conditioning and packing ----------------------------------------------------------------------------------

        private static Projection ProjectionOf(MiniMaxH3LoraTarget t) => t switch
        {
            MiniMaxH3LoraTarget.Q or MiniMaxH3LoraTarget.K or MiniMaxH3LoraTarget.V or MiniMaxH3LoraTarget.Qkv => Projection.Qkv,
            MiniMaxH3LoraTarget.Out => Projection.Out,
            MiniMaxH3LoraTarget.Fc1 => Projection.Fc1,
            _ => Projection.Fc2,
        };

        /// <summary>Read a term's factors as F32, check every value, fold its scale into up and swap diffusers' fc1 halves.</summary>
        private static (float[] Down, float[] Up) Condition(Term t, float strengthForMessage)
        {
            float[] down = Read(t.File, t.DownName, (long)t.Rank * t.In, t.FileName);
            float[] up = Read(t.File, t.UpName, t.Out * t.Rank, t.FileName);
            if (t.Scale != 1f)
            {
                LoraFactorMath.Scale(up, t.Scale);
                if (t.Scale > 1f && !LoraFactorMath.AllBelow(up, float.PositiveInfinity))
                    throw new InvalidDataException(
                        $"LoRA '{t.FileName}': strength {MiniMaxH3LoraConfig.Num(strengthForMessage)} scales {t.UpName} beyond the float range.");
            }
            if (t.Swap) MiniMaxH3LoraKeys.SwapHalves(up, t.Out, t.Rank);
            return (down, up);
        }

        private static float[] Read(SafetensorsFile st, string name, long count, string fileName)
        {
            var buffer = new float[count];
            st.ReadTensorDataToFloat32(st.GetInfo(name), buffer, count);
            for (long i = 0; i < count; i++)
                if (!float.IsFinite(buffer[i]))
                    throw new InvalidDataException(
                        $"LoRA '{fileName}': {name} holds a non-finite value ({buffer[i].ToString(CultureInfo.InvariantCulture)}) " +
                        $"at element {i}; the adapter file is damaged.");
            return buffer;
        }

        /// <summary>The pieces a term gives each slot of its projection: q / k / v their own slot; a fused qkv term its three
        /// thirds, each with the rank components whose up column is not zero on that third (exact: a zero column adds
        /// nothing); every other term the one slot.</summary>
        private void Slice(Term t, float[] down, float[] up, List<Piece>[] slots, bool threeSlots)
        {
            long inner = _cfg.AttentionInnerDim;
            switch (t.Target)
            {
                case MiniMaxH3LoraTarget.Q: slots[0].Add(new Piece(down, up, t.Rank)); return;
                case MiniMaxH3LoraTarget.K: slots[1].Add(new Piece(down, up, t.Rank)); return;
                case MiniMaxH3LoraTarget.V: slots[2].Add(new Piece(down, up, t.Rank)); return;
                case MiniMaxH3LoraTarget.Qkv when threeSlots:
                    for (int s = 0; s < 3; s++)
                    {
                        var keep = new List<int>();
                        for (int j = 0; j < t.Rank; j++)
                            for (long o = s * inner; o < (s + 1) * inner; o++)
                                if (up[o * t.Rank + j] != 0f)
                                {
                                    keep.Add(j);
                                    break;
                                }
                        if (keep.Count == 0) continue;
                        var d = new float[(long)keep.Count * t.In];
                        var u = new float[inner * keep.Count];
                        for (int k = 0; k < keep.Count; k++)
                        {
                            Array.Copy(down, (long)keep[k] * t.In, d, (long)k * t.In, t.In);
                            for (long o = 0; o < inner; o++) u[o * keep.Count + k] = up[(s * inner + o) * t.Rank + keep[k]];
                        }
                        slots[s].Add(new Piece(d, u, keep.Count));
                    }
                    return;
                default: slots[0].Add(new Piece(down, up, t.Rank)); return;
            }
        }

        /// <summary>Whether a fused qkv term reaches every third with every component (dense: one slot is cheaper than
        /// three copies of its down).</summary>
        private bool Dense(Term t, float[] up)
        {
            long inner = _cfg.AttentionInnerDim;
            for (int s = 0; s < 3; s++)
                for (int j = 0; j < t.Rank; j++)
                {
                    bool any = false;
                    for (long o = s * inner; o < (s + 1) * inner && !any; o++) any = up[o * t.Rank + j] != 0f;
                    if (!any) return false;
                }
            return true;
        }

        private static int Pad(int rank, int multiple) => (rank + multiple - 1) / multiple * multiple;

        private void Pack(List<Term> terms, List<SafetensorsFile> files)
        {
            // The safetensors maps are created on first use, which is not thread-safe: open them here.
            foreach (SafetensorsFile f in files)
                foreach (var info in f.Tensors.Values)
                {
                    f.TryGetTensorDataPointer(info, out _);
                    break;
                }
            var byProjection = terms.GroupBy(t => (t.Refiner, t.Block, Projection: ProjectionOf(t.Target))).ToArray();
            var options = new ParallelOptions { MaxDegreeOfParallelism = Math.Clamp(Environment.ProcessorCount, 1, MaxPackingThreads) };
            try
            {
                Parallel.ForEach(byProjection, options, group =>
                {
                    Factors packed = PackProjection(group.Key.Projection, group.ToList());
                    (group.Key.Refiner ? _refiner : _trunk)[group.Key.Block * ProjectionCount + (int)group.Key.Projection] = packed;
                });
            }
            catch (AggregateException e) when (e.InnerExceptions.Count > 0)
            {
                ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }
        }

        private Factors PackProjection(Projection projection, List<Term> terms)
        {
            var conditioned = terms.Select(t => (Term: t, Factors: Condition(t, _adapters[t.Adapter].Strength))).ToList();
            bool threeSlots = projection == Projection.Qkv &&
                              !conditioned.All(c => c.Term.Target == MiniMaxH3LoraTarget.Qkv && Dense(c.Term, c.Factors.Up));
            int slotCount = threeSlots ? 3 : 1;
            var slots = new List<Piece>[slotCount];
            for (int s = 0; s < slotCount; s++) slots[s] = new List<Piece>();
            foreach (var (term, (down, up)) in conditioned) Slice(term, down, up, slots, threeSlots);

            var (input, output) = projection switch
            {
                Projection.Qkv => MiniMaxH3LoraKeys.Shape(_cfg, MiniMaxH3LoraTarget.Qkv),
                Projection.Out => MiniMaxH3LoraKeys.Shape(_cfg, MiniMaxH3LoraTarget.Out),
                Projection.Fc1 => MiniMaxH3LoraKeys.Shape(_cfg, MiniMaxH3LoraTarget.Fc1),
                _ => MiniMaxH3LoraKeys.Shape(_cfg, MiniMaxH3LoraTarget.Fc2),
            };
            long slotRows = threeSlots ? _cfg.AttentionInnerDim : output;
            bool managed = Backend == BackendType.Cpu;
            bool f32Only = managed || Backend == BackendType.GgmlCpu;
            int multiple = managed ? 1 : Backend == BackendType.GgmlMetal ? 64 : 16;
            var parts = new List<Part>();
            for (int s = 0; s < slotCount; s++)
            {
                if (slots[s].Count == 0) continue;
                int rank = slots[s].Sum(p => p.Rank);
                // The pieces concatenated along the rank: down rows stacked, up columns side by side.
                var down = new float[(long)rank * input];
                var up = new float[slotRows * rank];
                int at = 0;
                foreach (Piece p in slots[s])
                {
                    Array.Copy(p.Down, 0, down, (long)at * input, (long)p.Rank * input);
                    for (long o = 0; o < slotRows; o++) Array.Copy(p.Up, o * p.Rank, up, o * rank + at, p.Rank);
                    at += p.Rank;
                }
                LoraFactorMath.Balance(down, up, rank, input, slotRows);
                bool f16 = !f32Only && LoraFactorMath.AllBelow(down, LoraFactorMath.F16Limit) && LoraFactorMath.AllBelow(up, LoraFactorMath.F16Limit);
                parts.Add(Store(s, down, up, rank, Pad(rank, multiple), input, slotRows, f16));
            }
            return new Factors { SlotRows = threeSlots ? slotRows : 0, Parts = parts.ToArray() };
        }

        /// <summary>down [rank, in] as ggml [in, padded] (zero rows after), up [out, rank] as ggml [padded, out] (each row's
        /// zero columns after).</summary>
        private unsafe Part Store(int slot, float[] down, float[] up, int rank, int padded, long input, long output, bool f16)
        {
            int elem = f16 ? 2 : 4;
            long downBytes = padded * input * elem, upBytes = output * padded * elem;
            IntPtr d = _buffers.Allocate(downBytes), u = _buffers.Allocate(upBytes);
            LoraFactorMath.StoreLoop(down.AsSpan(0, checked((int)((long)rank * input))), d, f16);
            if (padded > rank)
                System.Runtime.InteropServices.NativeMemory.Clear((byte*)d + (long)rank * input * elem, (nuint)((long)(padded - rank) * input * elem));
            if (padded == rank)
                LoraFactorMath.StoreLoop(up.AsSpan(0, checked((int)(output * rank))), u, f16);
            else
                for (long o = 0; o < output; o++)
                {
                    IntPtr row = u + (nint)(o * padded * elem);
                    LoraFactorMath.StoreLoop(up.AsSpan(checked((int)(o * rank)), rank), row, f16);
                    System.Runtime.InteropServices.NativeMemory.Clear((byte*)row + (long)rank * elem, (nuint)((long)(padded - rank) * elem));
                }
            Interlocked.Add(ref _factorBytes, downBytes + upBytes);
            Interlocked.Increment(ref f16 ? ref _f16Parts : ref _f32Parts);
            return new Part
            {
                Slot = slot, Format = f16 ? GraphFormat.F16 : GraphFormat.F32, Rank = padded, In = input, Out = output,
                Down = d, Up = u, DownBytes = downBytes, UpBytes = upBytes,
            };
        }

        // ---- reading back (tests, validation) --------------------------------------------------------------------------

        /// <summary>The effective update of a projection's output rows [<paramref name="row"/>, + <paramref name="count"/>), as
        /// the packed factors hold it (F16 or F32 read back): delta [count, in] row-major, in double, summed over the parts
        /// whose slot holds the rows. For checks against an independent merge.</summary>
        internal unsafe double[] DeltaRows(bool refiner, int block, Projection projection, long row, int count)
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                Factors f = Get(refiner, block, projection) ?? throw new InvalidOperationException("no adapter updates that projection");
                long input = f.Parts[0].In;
                var delta = new double[count * input];
                foreach (Part p in f.Parts)
                {
                    long first = f.SlotRows == 0 ? 0 : p.Slot * f.SlotRows;
                    for (int r = 0; r < count; r++)
                    {
                        long o = row + r - first;
                        if (o < 0 || o >= p.Out) continue;
                        for (int k = 0; k < p.Rank; k++)
                        {
                            double u = Value(p.Up, p.Format, o * p.Rank + k);
                            if (u == 0) continue;
                            for (long i = 0; i < input; i++) delta[r * input + i] += u * Value(p.Down, p.Format, k * input + i);
                        }
                    }
                }
                return delta;
            }
        }

        private static unsafe double Value(IntPtr data, int format, long index) =>
            format == GraphFormat.F16 ? (double)((System.Half*)data)[index] : ((float*)data)[index];

        // ---- report -----------------------------------------------------------------------------------------------------

        private string Describe(Adapter a)
        {
            string storage = $"{(_f32Parts == 0 ? "F16" : _f16Parts == 0 ? "F32" : "F16/F32")} factors" +
                             (Backend == BackendType.GgmlCpu ? " (ggml_cpu keeps F32)" : "") +
                             $", {(FactorBytes / 1e9).ToString("0.00", CultureInfo.InvariantCulture)} GB for the set";
            return $"  [lora] {a.WeightsFile} ({a.Name}{(a.ConfigFile != null ? "; " + a.ConfigFile : "; bare weights")}): " +
                   $"{(a.Contract != null ? "speed" : "style")} adapter, {MiniMaxH3LoraKeys.Spell(a.Grammar)} names, {a.Tensors} tensors, " +
                   $"{a.Terms} low-rank update(s) in {a.TrunkBlocks} transformer and {a.RefinerBlocks} token-refiner block(s), " +
                   $"rank {string.Join("/", a.Ranks)}, alpha from {a.AlphaSource}, strength {MiniMaxH3LoraConfig.Num(a.Strength)}, " +
                   $"scale {string.Join("/", a.Scales.Select(s => s.ToString("0.######", CultureInfo.InvariantCulture)))}; {storage}" +
                   (a.Contract != null
                       ? $"; samples {MiniMaxH3ImageLoraConfig.Describe(a.Contract)}" + (a.Strength != 1f ? "; the schedule was distilled at strength 1" : "")
                       : "");
        }

        // ---- lifetime ---------------------------------------------------------------------------------------------------

        /// <summary>Every factor pointer (the device caches key on them).</summary>
        internal IEnumerable<IntPtr> FactorPointers()
        {
            foreach (Factors[] table in new[] { _refiner, _trunk })
                foreach (Factors f in table)
                    if (f != null)
                        foreach (Part p in f.Parts)
                        {
                            yield return p.Down;
                            yield return p.Up;
                        }
        }

        /// <summary>Drop the device copies the GGML backends cached for the factors (bound by host pointer). The host factors
        /// stay; a later step uploads them again. A no-op after <see cref="Dispose"/>.</summary>
        internal void InvalidateDeviceCopies()
        {
            lock (_lock)
            {
                if (_disposed) return;
                InvalidateCore();
            }
        }

        private void InvalidateCore()
        {
            Action<IntPtr> hook = InvalidateHookForTest;
            foreach (IntPtr p in FactorPointers())
            {
                if (hook != null) hook(p);
                else if (Backend != BackendType.Cpu) GgmlBasicOps.InvalidateHostBuffer(p);
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                // Before the free: a backend must never keep a device copy keyed by a host address the allocator may reuse.
                InvalidateCore();
                _disposed = true;
                _buffers.Dispose();
            }
        }
    }
}
