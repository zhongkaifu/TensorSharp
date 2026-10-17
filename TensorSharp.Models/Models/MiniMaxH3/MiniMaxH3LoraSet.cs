// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A MiniMax-H3 LoRA plug-in (the LightX2V Turbo step-distillation adapters), loaded,
// validated and either packed for the native graph's unmerged update path
// (TSGH3DitAdapter, ggml_h3_lora.h) or kept open for the direct backends.
//
// Applied unmerged, y = W x + up (down x), as the reference applies it (inference_minimax_h3.py
// runs the adapter through PEFT and fuses only with --fuse-lora): the Q4_K / Q8_0 base stays
// as stored, and the update keeps its own factors. strength * alpha / rank is folded into up in
// F32 (alpha 8 at rank 128 is 0.0625; a loader assuming alpha = rank would apply 16x), then
// ff.net.0.proj's up halves are swapped into fc1's [gate | value] order (MiniMaxH3LoraKeys).
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>
    /// One MiniMax-H3 LoRA plug-in, selected by its <c>minimax-h3-lora</c> config.
    /// </summary>
    /// <remarks>
    /// <para><b>Accounting.</b> Every tensor in the file is used or the load fails naming it,
    /// with every other problem found in the same pass; nothing is skipped silently.</para>
    /// <para><b>Native packing.</b> Each rank component is balanced (down row and up column get
    /// equal norms, the product is unchanged) and stored as F16, unless some value would reach
    /// F16's range (then F32) or the backend is GgmlCpu, whose ARM kernels accumulate F16 dot
    /// products in F16 (F32 always there). Ranks are zero-padded to 64 on Metal, whose
    /// simdgroup matrix kernel needs K >= 64, and to 16 elsewhere. A block's q, k and v read one
    /// input, so their downs sit back to back in one allocation with one element type and the
    /// graph runs one stacked shrink; every other factor has its own 16 KiB-aligned allocation.
    /// Blocks pack in parallel, a bounded number at a time, each dropping its F32 working set
    /// as soon as it is stored.</para>
    /// <para><b>Direct backends.</b> The CPU, MLX and direct CUDA paths build their own exact
    /// F32 factors from <see cref="ReadFactors"/>, one resident copy each (<see cref="FactorBytes"/>
    /// once every block is built; on MLX a device array with no host mirror); the file stays open
    /// until <see cref="Dispose"/>, and each read checks its values there.</para>
    /// </remarks>
    internal sealed class MiniMaxH3LoraSet : IDisposable
    {
        private const int GgmlF32 = 0, GgmlF16 = 1;
        private const int MaxPackingThreads = 8;

        /// <summary>Test seam: when set, <see cref="InvalidateDeviceCopies"/> hands each factor
        /// pointer to it instead of <see cref="GgmlBasicOps.InvalidateHostBuffer"/>.</summary>
        internal static Action<IntPtr>? InvalidateHookForTest;

        private readonly LoraFactorBuffers _buffers = new();
        private readonly object _lock = new();
        private readonly MiniMaxH3Config _cfg;
        // One planned term per (block, target), null where the file has none:
        // index block * TargetCount + target.
        private readonly Term[] _trunk, _refiner;
        // The packed descriptors (native mode), mirrored into native memory for the adapter.
        private readonly H3BlockLora[] _trunkPacked, _refinerPacked;
        private SafetensorsFile _file;   // direct mode: open until Dispose
        private string _fileName;
        private long _factorBytes;
        private int _f16Groups, _f32Groups;
        private bool _disposed;

        private sealed class Term
        {
            internal string DownName, UpName;
            internal MiniMaxH3LoraTarget Target;
            internal int Rank;
            internal long In, Out;
            internal float Scale;   // strength * alpha / rank, folded into up
        }

        /// <summary>Per-worker F32 working arrays, grown on demand and reused block after block.</summary>
        private sealed class Scratch
        {
            internal readonly float[][] Down = new float[3][];
            internal readonly float[][] Up = new float[3][];
        }

        private MiniMaxH3LoraSet(MiniMaxH3Config cfg, bool direct)
        {
            _cfg = cfg;
            IsDirect = direct;
            _trunk = new Term[cfg.NumLayers * MiniMaxH3LoraKeys.TargetCount];
            _refiner = new Term[cfg.TokenRefinerLayers * MiniMaxH3LoraKeys.TargetCount];
            _trunkPacked = new H3BlockLora[cfg.NumLayers];
            _refinerPacked = new H3BlockLora[cfg.TokenRefinerLayers];
        }

        /// <summary>The sampling contract the plug-in locks.</summary>
        internal MiniMaxH3SamplingRecipe Recipe { get; private set; }

        /// <summary>The checkpoint partition the plug-in was written for (always FL2VA).</summary>
        internal MiniMaxH3Partition Partition { get; private set; }

        /// <summary>The adapter weights' full path.</summary>
        internal string WeightsPath { get; private set; }

        /// <summary>The strength applied: <c>--lora-scale</c>, else the config's <c>"scale"</c>, else 1.</summary>
        internal float Strength { get; private set; }

        /// <summary>The LoRA alpha in use (the file's metadata, or the config's when they agree or
        /// the file declares none).</summary>
        internal float Alpha { get; private set; }

        /// <summary>The one-line load report.</summary>
        internal string Summary { get; private set; } = "";

        /// <summary>Native bytes of the packed factors; in direct mode, the F32 bytes a full pass of
        /// <see cref="ReadFactors"/> hands out, which is also what the direct backend keeps resident
        /// once every block has run.</summary>
        internal long FactorBytes => Interlocked.Read(ref _factorBytes);

        /// <summary>The <see cref="H3DitAdapter"/> for <c>H3DitForwardArgs.Adapter</c>; zero in direct mode.</summary>
        internal IntPtr NativeAdapter { get; private set; }

        /// <summary>True when the set serves <see cref="ReadFactors"/> instead of a native adapter.</summary>
        internal bool IsDirect { get; }

        /// <summary>Tensors in the file (two per term).</summary>
        internal int TensorCount { get; private set; }

        /// <summary>Low-rank terms (one per updated projection).</summary>
        internal int TermCount { get; private set; }

        /// <summary>The distinct ranks in the file, ascending.</summary>
        internal IReadOnlyList<int> Ranks { get; private set; } = Array.Empty<int>();

        /// <summary>The distinct folded scales (strength * alpha / rank), ascending.</summary>
        internal IReadOnlyList<float> Scales { get; private set; } = Array.Empty<float>();

        // ---- loading ----------------------------------------------------------------------

        /// <summary>
        /// Load the one plug-in in <paramref name="resolved"/> (a spec after
        /// <see cref="LoraCliFlags.Resolve"/>: the weights, with the plug-in config as
        /// <see cref="LoraSpec.ConfigPath"/>) for a transformer of <paramref name="cfg"/>.
        /// </summary>
        /// <exception cref="ArgumentException">The spec is refused (bare weights, a foreign plug-in
        /// type, two plug-ins, weights the config was not written for, the wrong partition, a
        /// negative strength), with the canonical text.</exception>
        /// <exception cref="InvalidDataException">The config or the weights are malformed, or do
        /// not fit the transformer.</exception>
        /// <exception cref="FileNotFoundException">A named file is missing.</exception>
        internal static MiniMaxH3LoraSet Load(IReadOnlyList<LoraSpec> resolved, MiniMaxH3Config cfg, MiniMaxH3Partition partition,
            BackendType backend, bool direct)
        {
            ArgumentNullException.ThrowIfNull(resolved);
            ArgumentNullException.ThrowIfNull(cfg);
            if (resolved.Count == 0)
                throw new ArgumentException("MiniMax-H3 LoRA: no plug-in to load.", nameof(resolved));
            if (resolved.Count > 1)
                throw new ArgumentException(MiniMaxH3LoraConfig.TwoPluginsRefusal(
                    Path.GetFileName(resolved[0].Path), Path.GetFileName(resolved[1].Path)));
            var set = new MiniMaxH3LoraSet(cfg, direct);
            try
            {
                set.Open(resolved[0], partition, backend);
                return set;
            }
            catch
            {
                set.Dispose();
                throw;
            }
        }

        private void Open(LoraSpec spec, MiniMaxH3Partition partition, BackendType backend)
        {
            ArgumentNullException.ThrowIfNull(spec);
            string specFile = Path.GetFileName(spec.Path);

            // ---- the plug-in config ------------------------------------------------------------
            // DeclaredPluginType reports a missing --lora / --lora-config with the texts Resolve uses.
            string declared = LoraCliFlags.DeclaredPluginType(spec);
            string manifest = LoraCliFlags.ManifestPath(spec);
            if (declared == null || manifest == null)
                throw new ArgumentException(MiniMaxH3LoraConfig.BareWeightsRefusal(specFile));
            if (declared != MiniMaxH3LoraConfig.Type)
                throw new ArgumentException(MiniMaxH3LoraConfig.ForeignTypeRefusal(specFile, declared, "MiniMax-H3",
                    MiniMaxH3Model.ArchitectureId, MiniMaxH3LoraConfig.Type));
            var config = MiniMaxH3LoraConfig.Load(manifest);
            string manifestName = Path.GetFileName(config.Path);

            // A spec that still names the config itself (not run through Resolve) uses the
            // config's own weights entry, resolved without a download.
            string weights = spec.Path;
            if (string.Equals(Path.GetFullPath(spec.Path), config.Path, StringComparison.Ordinal))
            {
                // Resolve refuses this spelling too; a second config would otherwise be ignored.
                if (spec.ConfigPath != null)
                    throw new ArgumentException(
                        $"{LoraCliFlags.LoraFlag} {config.Path} is a plug-in config already; drop the {LoraCliFlags.ConfigFlag} after it.");
                weights = ConfigFileArgs.ResolveFileEntryPath(config.Path, "weights");
                if (weights == null || !File.Exists(weights))
                    throw new FileNotFoundException($"{LoraCliFlags.LoraFlag} file not found: {weights}", weights);
            }
            config.CheckWeights(weights);
            if (partition != config.Partition)
                throw new ArgumentException(MiniMaxH3LoraConfig.PartitionRefusal(manifestName, null));
            float strength = spec.Scale ?? config.Scale ?? 1f;
            MiniMaxH3LoraConfig.CheckStrength(config, weights, strength);

            Recipe = config.Recipe;
            Partition = config.Partition;
            WeightsPath = Path.GetFullPath(weights);
            Strength = strength;
            _fileName = Path.GetFileName(weights);

            var st = new SafetensorsFile(WeightsPath);
            bool keepOpen = false;
            try
            {
                // ---- the file's own declarations -------------------------------------------------
                st.Metadata.TryGetValue("key_format", out string keyFormat);
                // The ComfyUI twins state their layout as target_format instead.
                if (keyFormat == null && st.Metadata.TryGetValue("target_format", out string targetFormat) &&
                    targetFormat.Contains("ComfyUI", StringComparison.OrdinalIgnoreCase))
                    keyFormat = targetFormat;
                if (keyFormat != null && keyFormat != MiniMaxH3LoraConfig.KeyFormat)
                    throw new InvalidDataException(
                        $"{_fileName} has key_format '{keyFormat}'; {manifestName} expects '{MiniMaxH3LoraConfig.KeyFormat}' " +
                        "(the ComfyUI-format *_comfyui_bf16.safetensors files are not supported).");

                float? fileAlpha = null;
                if (st.Metadata.TryGetValue("alpha", out string rawAlpha))
                {
                    if (!float.TryParse(rawAlpha, NumberStyles.Float, CultureInfo.InvariantCulture, out float a) ||
                        !float.IsFinite(a) || !(a > 0f))
                        throw new InvalidDataException($"{_fileName}: __metadata__ \"alpha\" is '{rawAlpha}', not a positive number.");
                    fileAlpha = a;
                }
                if (config.Alpha is { } configured && fileAlpha is { } stated && configured != stated)
                    throw new InvalidDataException(
                        $"{manifestName} sets alpha {MiniMaxH3LoraConfig.Num(configured)}, but {_fileName}'s metadata says " +
                        $"{MiniMaxH3LoraConfig.Num(stated)}.");
                Alpha = config.Alpha ?? fileAlpha ?? throw new InvalidDataException(
                    $"{_fileName} declares no alpha (safetensors __metadata__ \"alpha\") and {manifestName} sets none; " +
                    "TensorSharp never assumes alpha = rank.");

                Plan(st);

                if (IsDirect)
                {
                    _file = st;
                    keepOpen = true;
                    foreach (var term in AllTerms())
                        _factorBytes += ((long)term.Rank * term.In + term.Out * term.Rank) * sizeof(float);
                }
                else
                {
                    Pack(st, backend);
                }
            }
            finally
            {
                if (!keepOpen) st.Dispose();
            }

            Summary = Describe(manifestName, config.Name, backend);
        }

        /// <summary>Group the file's tensors into terms, collecting every problem before failing.</summary>
        private void Plan(SafetensorsFile st)
        {
            var names = new List<string>(st.Tensors.Keys);
            names.Sort(StringComparer.Ordinal);
            TensorCount = names.Count;
            var rejected = new List<string>();
            var slots = new SortedSet<string>(StringComparer.Ordinal);
            var parts = new Dictionary<(bool Refiner, int Block, MiniMaxH3LoraTarget Target), (string A, string B)>();
            foreach (string name in names)
            {
                var key = MiniMaxH3LoraKeys.Resolve(name, _cfg.NumLayers, _cfg.TokenRefinerLayers, out string reason);
                if (key == null)
                {
                    rejected.Add($"{name}: {reason}");
                    continue;
                }
                var k = key.Value;
                slots.Add(k.Slot ?? "(none)");
                var id = (k.Refiner, k.Block, k.Target);
                parts.TryGetValue(id, out var pair);
                if (k.Up ? pair.B != null : pair.A != null)
                {
                    rejected.Add($"{name}: a second lora_{(k.Up ? "B" : "A")} for {MiniMaxH3LoraKeys.LoraModule(k.Refiner, k.Block, k.Target)}");
                    continue;
                }
                parts[id] = k.Up ? (pair.A, name) : (name, pair.B);
            }
            if (slots.Count > 1)
                rejected.Insert(0, $"several PEFT adapter slots ({string.Join(", ", slots)}); export the one to use on its own");

            var ranks = new SortedSet<int>();
            var scales = new SortedSet<float>();
            int terms = 0;
            // Refiner blocks first, the order the forward runs them in, so the report reads the same way.
            for (int pass = 0; pass < 2; pass++)
            {
                bool refiner = pass == 0;
                int blocks = refiner ? _cfg.TokenRefinerLayers : _cfg.NumLayers;
                for (int b = 0; b < blocks; b++)
                    for (int t = 0; t < MiniMaxH3LoraKeys.TargetCount; t++)
                    {
                        var target = (MiniMaxH3LoraTarget)t;
                        if (!parts.TryGetValue((refiner, b, target), out var pair)) continue;
                        string module = MiniMaxH3LoraKeys.LoraModule(refiner, b, target);
                        if (pair.A == null || pair.B == null)
                        {
                            rejected.Add($"{module}: has only lora_{(pair.A == null ? "B" : "A")}; a term needs both factors");
                            continue;
                        }
                        var (input, output) = MiniMaxH3LoraKeys.Shape(_cfg, target);
                        var a = st.GetInfo(pair.A);
                        var u = st.GetInfo(pair.B);
                        string bad = DtypeProblem(a) ?? DtypeProblem(u);
                        if (bad != null)
                        {
                            rejected.Add($"{module}: {bad}");
                            continue;
                        }
                        if (a.Shape.Length != 2 || u.Shape.Length != 2 || a.Shape[0] < 1 || a.Shape[0] > int.MaxValue ||
                            a.Shape[1] != input || u.Shape[0] != output || u.Shape[1] != a.Shape[0])
                        {
                            rejected.Add(
                                $"{module}: lora_A [{string.Join(",", a.Shape)}] / lora_B [{string.Join(",", u.Shape)}], but " +
                                $"{MiniMaxH3LoraKeys.GgufTarget(refiner, b, target, _cfg.AttentionInnerDim)} is {input} -> {output} " +
                                "(lora_A [rank, in], lora_B [out, rank]; a LoRA made for a different model)");
                            continue;
                        }
                        int rank = (int)a.Shape[0];
                        var term = new Term
                        {
                            DownName = pair.A, UpName = pair.B, Target = target, Rank = rank, In = input, Out = output,
                            Scale = Strength * Alpha / rank,
                        };
                        (refiner ? _refiner : _trunk)[b * MiniMaxH3LoraKeys.TargetCount + t] = term;
                        ranks.Add(rank);
                        scales.Add(term.Scale);
                        terms++;
                    }
            }

            if (rejected.Count > 0)
            {
                const int Shown = 6;
                var shown = rejected.Count <= Shown ? rejected : rejected.GetRange(0, Shown);
                throw new InvalidDataException(
                    $"LoRA '{_fileName}' does not fit MiniMax-H3 ({rejected.Count} problem(s)): {string.Join("; ", shown)}" +
                    (rejected.Count > Shown ? $"; ... and {rejected.Count - Shown} more" : "") +
                    ". Nothing is skipped silently: a partly applied LoRA would not be the adapter you asked for.");
            }
            if (terms == 0)
                throw new InvalidDataException($"LoRA '{_fileName}' contains no tensors.");
            TermCount = terms;
            Ranks = new List<int>(ranks);
            Scales = new List<float>(scales);
        }

        private static string DtypeProblem(SafetensorTensorInfo info) =>
            info.Dtype is SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32
                ? null
                : $"{info.Name} is {info.Dtype}; LoRA factors must be BF16, F16 or F32";

        private IEnumerable<Term> AllTerms()
        {
            foreach (var t in _refiner) if (t != null) yield return t;
            foreach (var t in _trunk) if (t != null) yield return t;
        }

        private Term TermAt(bool refiner, int block, MiniMaxH3LoraTarget target)
        {
            int blocks = refiner ? _cfg.TokenRefinerLayers : _cfg.NumLayers;
            if (block < 0 || block >= blocks)
                throw new ArgumentOutOfRangeException(nameof(block),
                    $"{(refiner ? "Refiner block" : "Block")} {block} is outside [0, {blocks}).");
            return (refiner ? _refiner : _trunk)[block * MiniMaxH3LoraKeys.TargetCount + (int)target];
        }

        // ---- conditioning -------------------------------------------------------------------
        // Plain loops (LoraFactorMath), for Mono.

        /// <summary>Read a term into <paramref name="down"/> / <paramref name="up"/> (grown when too
        /// small), check every value, fold the scale into up and swap fc1's halves.</summary>
        private void Condition(SafetensorsFile st, Term t, ref float[] down, ref float[] up)
        {
            long downCount = (long)t.Rank * t.In, upCount = t.Out * t.Rank;
            Read(st, t.DownName, ref down, downCount);
            Read(st, t.UpName, ref up, upCount);
            if (t.Scale != 1f)
            {
                var span = up.AsSpan(0, checked((int)upCount));
                LoraFactorMath.Scale(span, t.Scale);
                // A huge --lora-scale must not reach the graph as infinities.
                if (t.Scale > 1f && !LoraFactorMath.AllBelow(span, float.PositiveInfinity))
                    throw new InvalidDataException(
                        $"LoRA '{_fileName}': strength {MiniMaxH3LoraConfig.Num(Strength)} scales {t.UpName} beyond the float range.");
            }
            if (t.Target == MiniMaxH3LoraTarget.Fc1) MiniMaxH3LoraKeys.SwapHalves(up, t.Out, t.Rank);
        }

        private void Read(SafetensorsFile st, string name, ref float[] buffer, long count)
        {
            if (buffer == null || buffer.LongLength < count) buffer = new float[count];
            st.ReadTensorDataToFloat32(st.GetInfo(name), buffer, count);
            for (long i = 0; i < count; i++)
                if (!float.IsFinite(buffer[i]))
                    throw new InvalidDataException(
                        $"LoRA '{_fileName}': {name} holds a non-finite value ({buffer[i].ToString(CultureInfo.InvariantCulture)}) " +
                        $"at element {i}; the adapter file is damaged.");
        }

        /// <summary>
        /// The conditioned F32 factors of one term for the direct backends: down [rank, in], up
        /// [out, rank] with strength * alpha / rank folded in and fc1's halves in [gate | value]
        /// order. Null when the plug-in does not update that projection.
        /// </summary>
        /// <exception cref="InvalidOperationException">The set is packed for a native graph.</exception>
        /// <exception cref="InvalidDataException">The factors hold a non-finite value.</exception>
        internal (float[] Down, float[] Up, int Rank, long In, long Out)? ReadFactors(bool refiner, int block, MiniMaxH3LoraTarget t)
        {
            if (!IsDirect)
                throw new InvalidOperationException("ReadFactors serves the direct backends; this LoRA set is packed for the native graph.");
            var term = TermAt(refiner, block, t);
            if (term == null) return null;
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                float[] down = null, up = null;
                Condition(_file, term, ref down, ref up);
                return (down, up, term.Rank, term.In, term.Out);
            }
        }

        // ---- native packing -------------------------------------------------------------------

        private static int Pad(int rank, int multiple) => (rank + multiple - 1) / multiple * multiple;

        private unsafe void Pack(SafetensorsFile st, BackendType backend)
        {
            int multiple = backend == BackendType.GgmlMetal ? 64 : 16;
            bool f32Only = backend == BackendType.GgmlCpu;
            // The safetensors map is created on first use, which is not thread-safe: open it here.
            foreach (var term in AllTerms())
            {
                st.TryGetTensorDataPointer(st.GetInfo(term.DownName), out _);
                break;
            }

            int refinerBlocks = _cfg.TokenRefinerLayers, units = refinerBlocks + _cfg.NumLayers;
            var options = new ParallelOptions { MaxDegreeOfParallelism = Math.Clamp(Environment.ProcessorCount, 1, MaxPackingThreads) };
            try
            {
                Parallel.For(0, units, options, () => new Scratch(), (u, _, scratch) =>
                {
                    bool refiner = u < refinerBlocks;
                    int block = refiner ? u : u - refinerBlocks;
                    (refiner ? _refinerPacked : _trunkPacked)[block] = PackBlock(st, refiner, block, multiple, f32Only, scratch);
                    return scratch;
                }, _ => { });
            }
            catch (AggregateException e) when (e.InnerExceptions.Count > 0)
            {
                ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }

            IntPtr trunk = Mirror(_trunkPacked), refinerArray = Mirror(_refinerPacked);
            IntPtr adapter = _buffers.Allocate(sizeof(H3DitAdapter));
            *(H3DitAdapter*)adapter = new H3DitAdapter
            {
                StructBytes = sizeof(H3DitAdapter),
                NumBlocks = _cfg.NumLayers,
                NumRefinerBlocks = refinerBlocks,
                Blocks = trunk,
                Refiner = refinerArray,
            };
            NativeAdapter = adapter;
        }

        /// <summary>The descriptors in native memory, or zero for none.</summary>
        private unsafe IntPtr Mirror(H3BlockLora[] blocks)
        {
            if (blocks.Length == 0) return IntPtr.Zero;
            IntPtr p = _buffers.Allocate((long)blocks.Length * sizeof(H3BlockLora));
            blocks.AsSpan().CopyTo(new Span<H3BlockLora>((void*)p, blocks.Length));
            return p;
        }

        private H3BlockLora PackBlock(SafetensorsFile st, bool refiner, int block, int multiple, bool f32Only, Scratch scratch)
        {
            var result = new H3BlockLora();
            // q, k and v read one input: one allocation for their downs (the stacked shrink).
            var qkv = PackGroup(st, scratch, multiple, f32Only,
                TermAt(refiner, block, MiniMaxH3LoraTarget.Q), TermAt(refiner, block, MiniMaxH3LoraTarget.K),
                TermAt(refiner, block, MiniMaxH3LoraTarget.V));
            result.Q = qkv[0];
            result.K = qkv[1];
            result.V = qkv[2];
            result.Out = PackGroup(st, scratch, multiple, f32Only, TermAt(refiner, block, MiniMaxH3LoraTarget.Out))[0];
            result.Fc1 = PackGroup(st, scratch, multiple, f32Only, TermAt(refiner, block, MiniMaxH3LoraTarget.Fc1))[0];
            result.Fc2 = PackGroup(st, scratch, multiple, f32Only, TermAt(refiner, block, MiniMaxH3LoraTarget.Fc2))[0];
            return result;
        }

        /// <summary>
        /// Pack terms that read one input (null entries have no update): their downs back to back
        /// in one allocation, one element type for the group, each up in its own allocation.
        /// </summary>
        private H3Lora[] PackGroup(SafetensorsFile st, Scratch s, int multiple, bool f32Only, params Term[] group)
        {
            var packed = new H3Lora[group.Length];
            bool f16 = !f32Only;
            long totalDown = 0;
            for (int i = 0; i < group.Length; i++)
            {
                var t = group[i];
                if (t == null) continue;
                Condition(st, t, ref s.Down[i], ref s.Up[i]);
                LoraFactorMath.Balance(s.Down[i], s.Up[i], t.Rank, t.In, t.Out);
                // F16 unless some value would overflow it (then the whole group stays F32).
                f16 = f16 && LoraFactorMath.AllBelow(s.Down[i].AsSpan(0, checked((int)(t.Rank * t.In))), LoraFactorMath.F16Limit) &&
                      LoraFactorMath.AllBelow(s.Up[i].AsSpan(0, checked((int)(t.Out * t.Rank))), LoraFactorMath.F16Limit);
                totalDown += Pad(t.Rank, multiple) * t.In;
            }
            if (totalDown == 0) return packed;
            int elem = f16 ? 2 : 4;
            Interlocked.Increment(ref f16 ? ref _f16Groups : ref _f32Groups);
            IntPtr downBase = _buffers.Allocate(totalDown * elem);
            long cursor = 0;
            for (int i = 0; i < group.Length; i++)
            {
                var t = group[i];
                if (t == null) continue;
                int padded = Pad(t.Rank, multiple);
                IntPtr down = downBase + (nint)(cursor * elem);
                StoreDown(s.Down[i], t.Rank, padded, t.In, down, f16);
                cursor += padded * t.In;
                IntPtr up = _buffers.Allocate(t.Out * padded * elem);
                StoreUp(s.Up[i], t.Out, t.Rank, padded, up, f16);
                Interlocked.Add(ref _factorBytes, ((long)padded * t.In + t.Out * padded) * elem);
                packed[i] = new H3Lora
                {
                    Down = down, Up = up, Type = f16 ? GgmlF16 : GgmlF32, Rank = padded, In = t.In, Out = t.Out,
                };
            }
            return packed;
        }

        /// <summary>down [rank, in] as ggml [ne0 = in, ne1 = padded]: the rows, then zero rows.</summary>
        private static unsafe void StoreDown(float[] down, int rank, int padded, long input, IntPtr destination, bool f16)
        {
            int elem = f16 ? 2 : 4;
            long used = rank * input, total = padded * input;
            LoraFactorMath.StoreLoop(down.AsSpan(0, checked((int)used)), destination, f16);
            if (total > used)
                System.Runtime.InteropServices.NativeMemory.Clear((byte*)destination + used * elem, (nuint)((total - used) * elem));
        }

        /// <summary>up [out, rank] as ggml [ne0 = padded, ne1 = out]: each row, then its zero columns.</summary>
        private static unsafe void StoreUp(float[] up, long output, int rank, int padded, IntPtr destination, bool f16)
        {
            int elem = f16 ? 2 : 4;
            if (padded == rank)
            {
                LoraFactorMath.StoreLoop(up.AsSpan(0, checked((int)(output * rank))), destination, f16);
                return;
            }
            for (long o = 0; o < output; o++)
            {
                IntPtr row = destination + (nint)(o * padded * elem);
                LoraFactorMath.StoreLoop(up.AsSpan((int)(o * rank), rank), row, f16);
                System.Runtime.InteropServices.NativeMemory.Clear((byte*)row + (long)rank * elem, (nuint)((padded - rank) * elem));
            }
        }

        // ---- report -------------------------------------------------------------------------

        private string Describe(string manifestName, string name, BackendType backend)
        {
            int trunkBlocks = 0, refinerBlocks = 0;
            for (int b = 0; b < _cfg.NumLayers; b++)
                if (HasAny(_trunk, b)) trunkBlocks++;
            for (int b = 0; b < _cfg.TokenRefinerLayers; b++)
                if (HasAny(_refiner, b)) refinerBlocks++;
            string storage = IsDirect
                ? "exact F32 factors read per block for the direct backend"
                : $"{(_f32Groups == 0 ? "F16" : _f16Groups == 0 ? "F32" : "F16/F32")} factors" +
                  (backend == BackendType.GgmlCpu ? " (GgmlCpu keeps F32)" : "") +
                  $", rank padded to a multiple of {(backend == BackendType.GgmlMetal ? 64 : 16)}, " +
                  $"{(FactorBytes / 1e9).ToString("0.00", CultureInfo.InvariantCulture)} GB";
            return $"  [lora] {_fileName} ({name}; {manifestName}): {TermCount} low-rank update(s) in {trunkBlocks} transformer " +
                $"and {refinerBlocks} token-refiner block(s), rank {string.Join("/", Ranks)}, " +
                $"alpha {MiniMaxH3LoraConfig.Num(Alpha)}, strength {MiniMaxH3LoraConfig.Num(Strength)}, " +
                $"scale {string.Join("/", FormatScales())}; {storage}" +
                (Strength != 1f ? "; the schedule was distilled at strength 1" : "");
        }

        private static bool HasAny(Term[] terms, int block)
        {
            for (int t = 0; t < MiniMaxH3LoraKeys.TargetCount; t++)
                if (terms[block * MiniMaxH3LoraKeys.TargetCount + t] != null) return true;
            return false;
        }

        private IEnumerable<string> FormatScales()
        {
            foreach (float s in Scales) yield return s.ToString("0.######", CultureInfo.InvariantCulture);
        }

        // ---- lifetime -------------------------------------------------------------------------

        /// <summary>
        /// Drop the device copies the GGML backends cached for the packed factors (they are bound
        /// by host pointer). The host factors stay; a later forward re-uploads them. A no-op in
        /// direct mode and after <see cref="Dispose"/>.
        /// </summary>
        internal void InvalidateDeviceCopies()
        {
            lock (_lock)
            {
                if (_disposed) return;
                InvalidateDeviceCopiesCore();
            }
        }

        private void InvalidateDeviceCopiesCore()
        {
            var pointers = new List<IntPtr>();
            var seen = new HashSet<IntPtr>();
            void Add(IntPtr p)
            {
                if (p != IntPtr.Zero && seen.Add(p)) pointers.Add(p);
            }
            foreach (var blocks in new[] { _refinerPacked, _trunkPacked })
                foreach (var b in blocks)
                    foreach (var l in new[] { b.Q, b.K, b.V, b.Out, b.Fc1, b.Fc2 })
                    {
                        Add(l.Down);
                        Add(l.Up);
                    }
            var hook = InvalidateHookForTest;
            foreach (var p in pointers)
            {
                if (hook != null) hook(p);
                else GgmlBasicOps.InvalidateHostBuffer(p);
            }
        }

        /// <summary>The packed up factor of one term read back as F32 [out, padded rank]: the
        /// rows the native graph multiplies, after the scale fold, the fc1 swap and balancing.</summary>
        internal unsafe float[] PackedUpForTest(bool refiner, int block, MiniMaxH3LoraTarget t)
        {
            if (IsDirect) throw new InvalidOperationException("A direct-mode set has no packed factors.");
            if (TermAt(refiner, block, t) == null) throw new InvalidOperationException("The plug-in does not update that projection.");
            var b = (refiner ? _refinerPacked : _trunkPacked)[block];
            var l = t switch
            {
                MiniMaxH3LoraTarget.Q => b.Q,
                MiniMaxH3LoraTarget.K => b.K,
                MiniMaxH3LoraTarget.V => b.V,
                MiniMaxH3LoraTarget.Out => b.Out,
                MiniMaxH3LoraTarget.Fc1 => b.Fc1,
                _ => b.Fc2,
            };
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                int count = checked((int)(l.Out * l.Rank));
                var result = new float[count];
                if (l.Type == GgmlF16)
                {
                    var source = new ReadOnlySpan<System.Half>((void*)l.Up, count);
                    for (int i = 0; i < count; i++) result[i] = (float)source[i];
                }
                else
                {
                    new ReadOnlySpan<float>((void*)l.Up, count).CopyTo(result);
                }
                return result;
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                // Before the free: a backend must never keep a device copy keyed by a host
                // address the allocator may hand out again.
                InvalidateDeviceCopiesCore();
                _disposed = true;
                _file?.Dispose();
                _file = null;
                _buffers.Dispose();
                NativeAdapter = IntPtr.Zero;
            }
        }
    }
}
