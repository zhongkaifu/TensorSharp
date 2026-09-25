// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage;

/// <summary>
/// The LoRA plug-ins of one Qwen-Image-2.1 transformer, loaded, validated and packed for
/// the native graph's unmerged update path (<see cref="QwenImage21Adapter"/>).
/// </summary>
/// <remarks>
/// <para><b>Why unmerged.</b> A step-distillation LoRA moves the weights by about 0.1-0.5%,
/// the size of Q8_0's own rounding step: dequantize, add and requantize loses most of it
/// (measured on the Pruna adapter: cosine 0.07 between the surviving and the intended
/// delta in block 0). The base stays as stored and each projection adds up * (down * x).</para>
/// <para><b>Packing.</b> Scales (strength, alpha / rank, DoRA magnitude) are folded into
/// <c>up</c> in F32. Each rank component is then rebalanced (down row and up column get
/// equal norms, the product is unchanged) so F16 keeps both factors well inside its range;
/// a group that would overflow F16 stays F32. Several LoRAs on one projection are
/// concatenated along the rank. The ranks are zero-padded (to 64 on Metal, whose simdgroup
/// matrix kernel needs K >= 64, 16 elsewhere). Q, K and V (and the gate and up halves) read
/// one input, so their down factors share one allocation and the graph runs one shrink.</para>
/// <para><b>Accounting.</b> Every tensor in a file is used or the load fails and names it;
/// nothing is skipped silently.</para>
/// </remarks>
internal sealed class QwenImage21LoraSet : IDisposable
{
    private const int GgmlF32 = 0, GgmlF16 = 1;
    private const int Channels = QwenImage21DiT.Channels, Dim = QwenImage21DiT.HiddenSize;
    private const long PageAlignment = 16384;
    private const string PddFormatPrefix = "qwenimage21_extracted_prefused";

    private readonly List<IntPtr> _buffers = new();
    private IntPtr[] _adapters = Array.Empty<IntPtr>();

    /// <summary>The sampling recipe of the plug-in that carries one, or null.</summary>
    internal QwenImage21LoraRecipe Recipe { get; private set; }
    /// <summary>Blocks whose fused gate_up weight must be described as its two halves.</summary>
    internal bool[] SplitGateUp { get; } = new bool[QwenImage21DiT.Layers];
    /// <summary>Replacement F32 gains (a PDD bundle), zero where the checkpoint's own is kept.</summary>
    internal IntPtr[] NormQ { get; } = new IntPtr[QwenImage21DiT.Layers];
    internal IntPtr[] NormK { get; } = new IntPtr[QwenImage21DiT.Layers];
    internal IntPtr TextNorm { get; private set; }
    /// <summary>Per-step output heads (F32 [dim, channels]) replacing proj_out, or empty.</summary>
    internal IntPtr[] OutputHeads { get; private set; } = Array.Empty<IntPtr>();
    internal string Summary { get; private set; } = "";
    private long _factorBytes;
    internal long FactorBytes => Interlocked.Read(ref _factorBytes);

    // ---- building ---------------------------------------------------------------------

    /// <summary>One low-rank term: down [rank, in], up [out, rank] with every scale applied.
    /// Two halves of a fused gate_up LoRA share the same <see cref="Down"/> array.</summary>
    private sealed class Term
    {
        internal float[] Down, Up;
        internal int Rank;
        internal long In, Out;
    }

    /// <summary>All updates of one projection, in plug-in order.</summary>
    private sealed class Slot
    {
        internal readonly List<Term> Terms = new();
        internal float[] RowScale;   // DoRA; applies to the base and to earlier terms
        internal long In, Out;
        internal bool Empty => Terms.Count == 0 && RowScale == null;
    }

    private readonly Dictionary<string, Slot> _slots = new(StringComparer.Ordinal);

    private Slot SlotFor(string name, long input, long output)
    {
        if (!_slots.TryGetValue(name, out var slot)) _slots[name] = slot = new Slot { In = input, Out = output };
        return slot;
    }

    /// <summary>
    /// Load <paramref name="specs"/> against the transformer GGUF <paramref name="dit"/>
    /// (tensor names prefixed by <paramref name="prefix"/>). <paramref name="ranks"/> is the
    /// tensor-parallel degree the transformer will shard over (1 = one device).
    /// </summary>
    internal static QwenImage21LoraSet Load(IReadOnlyList<LoraSpec> specs, GgufFile dit, string prefix,
        BackendType backend, int ranks)
    {
        var set = new QwenImage21LoraSet();
        try
        {
            var lines = new List<string>();
            foreach (var spec in specs) lines.Add(set.AddFile(spec, dit, prefix));
            set.Pack(dit, prefix, backend == BackendType.GgmlMetal ? 64 : 16, Math.Max(1, ranks));
            set.Summary = string.Join(Environment.NewLine, lines);
            return set;
        }
        catch
        {
            set.Dispose();
            throw;
        }
    }

    private static (long In, long Out) WeightShape(GgufFile dit, string prefix, string module)
    {
        if (!dit.Tensors.TryGetValue(prefix + module + ".weight", out var info) || info.Shape.Length != 2)
            return (-1, -1);
        return ((long)info.Shape[0], (long)info.Shape[1]);
    }

    private string AddFile(LoraSpec spec, GgufFile dit, string prefix)
    {
        string file = Path.GetFileName(spec.Path);
        using var st = new SafetensorsFile(spec.Path);

        // ---- companion configuration -----------------------------------------------------
        st.Metadata.TryGetValue("format", out string format);
        bool pddFile = format != null && format.StartsWith(PddFormatPrefix, StringComparison.Ordinal);
        QwenImage21LoraConfig config = null;
        string configNote = "";
        string dir = Path.GetDirectoryName(spec.Path) ?? ".";
        string peft = Path.Combine(dir, "adapter_config.json");
        // A PEFT save_pretrained folder keeps alpha only in adapter_config.json, beside a file
        // with this exact name; it stays an alpha source when another config is named too.
        bool peftFolder = file.Equals("adapter_model.safetensors", StringComparison.OrdinalIgnoreCase) && File.Exists(peft);
        QwenImage21LoraConfig folder = null;
        if (spec.ConfigPath != null)
        {
            config = QwenImage21LoraConfig.Load(spec.ConfigPath);
            configNote = $"config {Path.GetFileName(config.Path)} ({config.Format})";
            if (peftFolder && !string.Equals(Path.GetFullPath(peft), config.Path, StringComparison.Ordinal))
            {
                folder = QwenImage21LoraConfig.Load(peft);
                configNote += $"; alpha source {peft} (PEFT folder)";
            }
        }
        else
        {
            string pdd = Path.Combine(dir, "pdd_config.json");
            if (pddFile)
            {
                // The bundle is not usable without its trained grid; its reference loader
                // reads pdd_config.json from the same folder, and so does this one.
                if (!File.Exists(pdd))
                    throw new FileNotFoundException(
                        $"LoRA '{file}' is a VideoX-Fun parallel-decoding bundle ({format}); it needs its pdd_config.json " +
                        $"(sigma grid, alpha). Place it next to the weights or pass it with --lora-config.", pdd);
                config = QwenImage21LoraConfig.Load(pdd);
                configNote = $"config {pdd} (found next to the weights)";
            }
            else if (peftFolder)
            {
                config = QwenImage21LoraConfig.Load(peft);
                configNote = $"config {peft} (PEFT folder)";
            }
        }
        if (pddFile && config is { IsPdd: false })
            throw new InvalidDataException($"LoRA '{file}' is a PDD bundle but its config {config.Path} is {config.Format}, not a pdd_config.json.");
        if (config is { IsPdd: true } && !pddFile)
            throw new InvalidDataException($"{config.Path} is a PDD config, but '{file}' does not declare the PDD bundle format.");

        QwenImage21LoraConfig embedded = null;
        if (st.Metadata.TryGetValue("lora_adapter_metadata", out var peftJson))
            embedded = QwenImage21LoraConfig.FromAdapterMetadata(peftJson, spec.Path);
        // Training metadata, not an alpha source: kohya files carry a per-module .alpha, and a
        // converter that dropped those folded alpha into the factors (ComfyUI, diffusers and
        // sd-scripts all ignore this key). Read only to say so when it disagrees.
        float? kohyaAlpha = st.Metadata.TryGetValue("ss_network_alpha", out var ka) &&
            float.TryParse(ka, NumberStyles.Float, CultureInfo.InvariantCulture, out float kav) ? kav : null;
        bool anyAlphaTensor = false, kohyaDisagrees = false;
        if (config?.Alpha is { } ca && embedded?.Alpha is { } ea && ca != ea)
            Console.WriteLine($"  [lora] {file}: {config.Path} sets alpha {ca}, the file's own metadata {ea}; using {ca} (explicit config wins).");
        if (config == null && embedded != null) configNote = "alpha from the file's PEFT metadata";

        if (config?.Recipe != null)
        {
            if (Recipe != null)
                throw new ArgumentException(
                    $"Two LoRA plug-ins define a sampling recipe ({Recipe.Source} and {config.Recipe.Source}). A step-distilled " +
                    "LoRA is trained for its own schedule; stacking two of them is not meaningful. Keep one recipe.");
            Recipe = config.Recipe;
        }
        float strength = spec.Scale ?? config?.Scale ?? 1f;
        if (!float.IsFinite(strength)) throw new ArgumentException($"LoRA '{file}': the strength must be finite.");

        // ---- group the tensors by module ---------------------------------------------------
        var modules = new Dictionary<string, Dictionary<QwenImage21LoraPart, string>>(StringComparer.Ordinal);
        var rejected = new List<string>();
        var slots = new HashSet<string>(StringComparer.Ordinal);
        foreach (var name in st.Tensors.Keys.OrderBy(n => n, StringComparer.Ordinal))
        {
            var key = QwenImage21LoraKeys.Resolve(name, out string reason, out string adapterSlot);
            if (key == null) { rejected.Add($"{name}: {reason}"); continue; }
            if (adapterSlot != null) slots.Add(adapterSlot);
            if (!modules.TryGetValue(key.Value.Module, out var parts)) modules[key.Value.Module] = parts = new();
            if (parts.ContainsKey(key.Value.Part))
            {
                rejected.Add($"{name}: a second {key.Value.Part} tensor for {key.Value.Module}");
                continue;
            }
            parts[key.Value.Part] = name;
        }
        if (slots.Count > 1)
            throw new InvalidDataException(
                $"LoRA '{file}' holds several PEFT adapters ({string.Join(", ", slots)}); export the one to use on its own.");
        if (rejected.Count > 0)
            throw new InvalidDataException(
                $"LoRA '{file}' has {rejected.Count} tensor(s) TensorSharp cannot apply to Qwen-Image-2.1, e.g. " +
                string.Join("; ", rejected.Take(4)) + (rejected.Count > 4 ? "; ..." : "") +
                ". Nothing is skipped silently: a partly applied LoRA would not be the adapter you asked for.");
        if (modules.Count == 0)
            throw new InvalidDataException($"LoRA '{file}' contains no tensors.");

        // ---- per module --------------------------------------------------------------------
        // Validation, alpha and the replaced parameters run in module order. Reading and
        // conditioning the factors, the bulk of a load, runs in parallel. The updates then
        // reach their slots in module order again, so the result does not depend on scheduling.
        bool fusedCheckpoint = dit.Tensors.ContainsKey(prefix + "transformer_blocks.0.img_mlp.gate_up.weight");
        int terms = 0, doras = 0, replaced = 0;
        var ranks = new SortedSet<int>();
        var scales = new SortedSet<float>();
        var updates = new List<Update>();
        foreach (var (module, parts) in modules.OrderBy(m => m.Key, StringComparer.Ordinal))
        {
            int block = QwenImage21LoraKeys.BlockOf(module, out string local);
            if (parts.TryGetValue(QwenImage21LoraPart.Weight, out string full) || parts.TryGetValue(QwenImage21LoraPart.Diff, out _))
            {
                if (parts.Count != 1)
                    throw new InvalidDataException($"LoRA '{file}': {module} mixes a full value with low-rank factors.");
                replaced += Replace(st, module, block, local, parts, pddFile ? config : null, file, dit, prefix, strength);
                continue;
            }
            parts.TryGetValue(QwenImage21LoraPart.Down, out string downName);
            parts.TryGetValue(QwenImage21LoraPart.Up, out string upName);
            if ((downName == null) != (upName == null))
                throw new InvalidDataException($"LoRA '{file}': {module} has only one of its two factors.");
            if (parts.ContainsKey(QwenImage21LoraPart.Alpha) && downName == null)
                throw new InvalidDataException($"LoRA '{file}': {module} has an alpha but no factors.");

            // Target geometry: gate_layer / proj are the halves of a fused gate_up.
            string target = module;
            long halfOffset = -1;
            if (local is "img_mlp.gate_layer" or "img_mlp.proj" && fusedCheckpoint)
            {
                target = $"transformer_blocks.{block}.img_mlp.gate_up";
                halfOffset = local == "img_mlp.gate_layer" ? 0 : 1;
            }
            var (input, output) = WeightShape(dit, prefix, target);
            if (input < 0 && local == "img_mlp.gate_up" && !fusedCheckpoint)
            {
                var (gi, go) = WeightShape(dit, prefix, $"transformer_blocks.{block}.img_mlp.gate_layer");
                (input, output) = (gi, go * 2);
            }
            if (input < 0)
                throw new InvalidDataException($"LoRA '{file}' targets {module}, which the transformer GGUF does not have.");
            if (halfOffset >= 0) output /= 2;

            // Slots: a fused gate_up LoRA splits into halves sharing their down factor.
            string[] slotNames = block < 0 ? new[] { module }
                : local == "img_mlp.gate_up" ? new[] { $"{block}.gate", $"{block}.up" }
                : new[]
                {
                    $"{block}." + local switch
                    {
                        "attn.to_q" => "q", "attn.to_k" => "k", "attn.to_v" => "v", "attn.to_out.0" => "out",
                        "img_mlp.gate_layer" => "gate", "img_mlp.proj" => "up", "img_mlp.out" => "down",
                        _ => throw new InvalidDataException($"LoRA '{file}': unexpected module {module}."),
                    },
                };
            var update = new Update { Slots = slotNames, In = input, Out = output };

            if (parts.TryGetValue(QwenImage21LoraPart.DoraScale, out string doraName))
            {
                var shape = st.GetInfo(doraName).Shape;
                if (!(shape.Length == 2 && shape[0] == output && shape[1] == 1) && !(shape.Length == 1 && shape[0] == output))
                    throw new NotSupportedException(
                        $"LoRA '{file}': {doraName} has shape [{string.Join(",", shape)}]; only output-axis DoRA magnitudes ([out, 1]) are supported.");
                update.DoraName = doraName;
                // Here rather than in the parallel pass: the GGUF reader is not thread-safe
                // (RowNorms parallelizes internally).
                update.BaseNorms = BaseRowNorms(dit, prefix, module, block, local, fusedCheckpoint, output);
                doras++;
            }

            if (downName != null)
            {
                var a = st.GetInfo(downName).Shape;
                var b = st.GetInfo(upName).Shape;
                if (a.Length != 2 || b.Length != 2)
                    throw new NotSupportedException($"LoRA '{file}': {module} factors are not 2-D (convolution LoRAs do not apply to this transformer).");
                long rank = a[0];
                if (a[1] != input || b[0] != output || b[1] != rank)
                    throw new InvalidDataException(
                        $"LoRA '{file}': {module} factors are down [{a[0]},{a[1]}] / up [{b[0]},{b[1]}], but the projection is " +
                        $"{input} -> {output}. This LoRA was made for a different model.");
                bool hasAlpha = parts.TryGetValue(QwenImage21LoraPart.Alpha, out string alphaName);
                anyAlphaTensor |= hasAlpha;
                float? configured = config?.AlphaFor(module) ?? folder?.AlphaFor(module) ?? embedded?.AlphaFor(module);
                float alpha = hasAlpha ? st.ReadFloat32(alphaName)[0] : configured ?? rank;
                kohyaDisagrees |= !hasAlpha && configured == null && kohyaAlpha is { } k && k != rank;
                bool rs = config?.UseRsLora ?? folder?.UseRsLora ?? embedded?.UseRsLora ?? false;
                float scale = strength * alpha / (rs ? MathF.Sqrt(rank) : rank);
                ranks.Add((int)rank);
                scales.Add(scale);
                (update.DownName, update.UpName, update.Rank, update.Scale) = (downName, upName, (int)rank, scale);
                terms++;
            }
            updates.Add(update);
        }

        // The safetensors map is created on first use, which is not thread-safe: open it here.
        if (updates.Count > 0) st.TryGetTensorDataPointer(st.GetInfo(updates[0].DownName ?? updates[0].DoraName), out _);
        Parallel.ForEach(updates, u => Condition(u, st, strength));
        foreach (var u in updates)
            for (int i = 0; i < u.Slots.Length; i++)
                AddToSlot(SlotFor(u.Slots[i], u.In, u.Out / u.Slots.Length), u.Terms[i], u.RowScales[i]);

        if (kohyaDisagrees && !anyAlphaTensor)
            Console.WriteLine($"  [lora] {file}: ignoring ss_network_alpha={kohyaAlpha} (training metadata); without .alpha tensors " +
                "alpha = rank, as in ComfyUI and diffusers. Pass a --lora-config with \"alpha\" if this file needs it.");
        if (pddFile)
        {
            if (OutputHeads.Length == 0)
                throw new InvalidDataException($"LoRA '{file}' is a PDD bundle without its per-step proj_out heads.");
            if (strength != 1f)
                Console.WriteLine($"  [lora] {file}: strength {strength} scales the low-rank deltas only; the bundle's step heads and norm gains are full replacements.");
        }
        return $"  [lora] {file}: {terms} low-rank update(s)" +
            (ranks.Count > 0 ? $", rank {string.Join("/", ranks)}" : "") +
            (scales.Count > 0 ? $", scale {string.Join("/", scales.Select(s => s.ToString("0.####", CultureInfo.InvariantCulture)))}" : "") +
            (doras > 0 ? $", {doras} DoRA magnitude(s)" : "") +
            (replaced > 0 ? $", {replaced} replaced parameter(s)" : "") +
            (OutputHeads.Length > 0 && pddFile ? $", {OutputHeads.Length} per-step output heads" : "") +
            (configNote.Length > 0 ? $"; {configNote}" : "");
    }

    /// <summary>One module's update between planning and routing.</summary>
    private sealed class Update
    {
        internal string[] Slots;            // one slot, or the gate and up halves
        internal long In, Out;              // Out spans both halves when split
        internal string DownName, UpName, DoraName;
        internal int Rank;
        internal float Scale;
        internal float[] BaseNorms;         // checkpoint row norms, for a DoRA
        internal Term[] Terms;              // per slot, from Condition
        internal float[][] RowScales;
    }

    /// <summary>Read one module's factors and apply its scale, DoRA magnitude and balancing.
    /// Touches only <paramref name="u"/> and the (already mapped) safetensors file.</summary>
    private static void Condition(Update u, SafetensorsFile st, float strength)
    {
        float[] rowScale = u.DoraName != null ? st.ReadFloat32(u.DoraName) : null;
        Term term = null;
        if (u.DownName != null)
        {
            term = new Term { Down = st.ReadFloat32(u.DownName), Up = st.ReadFloat32(u.UpName), Rank = u.Rank, In = u.In, Out = u.Out };
            if (u.Scale != 1f) TensorPrimitives.Multiply(term.Up, u.Scale, term.Up);
        }

        if (rowScale != null)
        {
            // ComfyUI weight_decompose on the output axis: W' = W + s * (g * (W + dW) - W),
            // g = magnitude / ||W_row|| of the checkpoint's weight. In runtime form the base
            // rows scale by 1 + s * (g - 1) and the low-rank term by s * g (dW already
            // carries s through its scale, so it takes g only).
            var baseScale = new float[u.Out];
            for (long o = 0; o < u.Out; o++)
            {
                // ComfyUI adds torch.finfo(dtype).eps; F32's keeps the magnitude exact.
                float g = rowScale[o] / (u.BaseNorms[o] + 1.1920929e-7f);
                baseScale[o] = 1f + strength * (g - 1f);
                rowScale[o] = g;
            }
            if (term != null)
                for (long o = 0; o < u.Out; o++)
                    TensorPrimitives.Multiply(term.Up.AsSpan((int)(o * term.Rank), term.Rank), rowScale[o], term.Up.AsSpan((int)(o * term.Rank), term.Rank));
            rowScale = baseScale;
        }
        if (term != null) Balance(term);

        if (u.Slots.Length == 1)
        {
            u.Terms = new[] { term };
            u.RowScales = new[] { rowScale };
            return;
        }
        long half = u.Out / 2;
        Term Half(int i) => term == null ? null : new Term
        {
            Down = term.Down, Rank = term.Rank, In = u.In, Out = half,
            Up = term.Up.AsSpan((int)(i * half * term.Rank), (int)(half * term.Rank)).ToArray(),
        };
        u.Terms = new[] { Half(0), Half(1) };
        u.RowScales = new[] { rowScale?.AsSpan(0, (int)half).ToArray(), rowScale?.AsSpan((int)half).ToArray() };
    }

    private static void AddToSlot(Slot slot, Term term, float[] rowScale)
    {
        if (rowScale != null)
        {
            // A DoRA applied after earlier updates rescales them too (ComfyUI applies
            // patches in order, each on the weight the previous ones produced). Its g is
            // taken from the checkpoint's row norms: exact for a lone DoRA; after other
            // updates of the same projection ComfyUI would use the patched weight's norms,
            // which differ by the (tiny) relative size of those updates.
            slot.RowScale ??= Enumerable.Repeat(1f, (int)slot.Out).ToArray();
            for (int o = 0; o < slot.Out; o++) slot.RowScale[o] *= rowScale[o];
            foreach (var earlier in slot.Terms)
                for (int o = 0; o < slot.Out; o++)
                    TensorPrimitives.Multiply(earlier.Up.AsSpan(o * earlier.Rank, earlier.Rank), rowScale[o], earlier.Up.AsSpan(o * earlier.Rank, earlier.Rank));
        }
        if (term != null) slot.Terms.Add(term);
    }

    /// <summary>Give every rank component equal down-row and up-column norms; the product is unchanged.</summary>
    private static void Balance(Term t)
    {
        var upNorm = new double[t.Rank];
        for (long o = 0; o < t.Out; o++)
            for (int r = 0; r < t.Rank; r++) { double v = t.Up[o * t.Rank + r]; upNorm[r] += v * v; }
        var divisors = new float[t.Rank];
        for (int r = 0; r < t.Rank; r++)
        {
            divisors[r] = 1f;
            double down = TensorPrimitives.Norm(t.Down.AsSpan((int)(r * t.In), (int)t.In));
            double up = Math.Sqrt(upNorm[r]);
            if (down <= 0 || up <= 0) continue;
            float k = (float)Math.Sqrt(up / down);
            TensorPrimitives.Multiply(t.Down.AsSpan((int)(r * t.In), (int)t.In), k, t.Down.AsSpan((int)(r * t.In), (int)t.In));
            divisors[r] = k;
        }
        // Row by row: the up factor is [out, rank], so a column at a time would stride.
        for (long o = 0; o < t.Out; o++)
            TensorPrimitives.Divide(t.Up.AsSpan((int)(o * t.Rank), t.Rank), divisors, t.Up.AsSpan((int)(o * t.Rank), t.Rank));
    }

    /// <summary>Row norms of the checkpoint weight a DoRA magnitude normalizes: gate_layer and
    /// proj are the halves of a fused gate_up, and a fused gate_up LoRA spans both halves of
    /// an unfused checkpoint.</summary>
    private static float[] BaseRowNorms(GgufFile dit, string prefix, string module, int block, string local, bool fused, long rows)
    {
        string mlp = $"{prefix}transformer_blocks.{block}.img_mlp.";
        if (fused && local is "img_mlp.gate_layer" or "img_mlp.proj")
            return RowNorms(dit, mlp + "gate_up.weight", local == "img_mlp.proj" ? rows : 0, rows);
        if (!fused && local == "img_mlp.gate_up")
            return RowNorms(dit, mlp + "gate_layer.weight", 0, rows / 2).Concat(RowNorms(dit, mlp + "proj.weight", 0, rows / 2)).ToArray();
        return RowNorms(dit, prefix + module + ".weight", 0, rows);
    }

    /// <summary>Euclidean norms of rows [first, first + count) of a GGUF weight, dequantized.</summary>
    private static float[] RowNorms(GgufFile dit, string tensor, long first, long count)
    {
        var info = dit.Tensors[tensor];
        long input = (long)info.Shape[0], rows = (long)info.Shape[1];
        long rowBytes = dit.GetTensorByteCount(info) / rows;
        IntPtr data;
        IntPtr owned = IntPtr.Zero;
        if (!dit.TryGetTensorDataPointer(info, out data))
        {
            owned = QuantizedWeight.AllocateBuffer(dit.GetTensorByteCount(info));
            dit.ReadTensorDataToNative(info, owned, dit.GetTensorByteCount(info));
            data = owned;
        }
        try
        {
            var result = new float[count];
            Parallel.For(0, (count + 255) / 256, chunk =>
            {
                long start = chunk * 256, n = Math.Min(256, count - start);
                // One row at a time: a buffer for all 256 rows is a large-object allocation per
                // chunk, and a DoRA file's thousands of them ran back-to-back gen-2 collections.
                var row = new float[input];
                for (long i = 0; i < n; i++)
                {
                    NativeDequant.DequantizeToFloat32((int)info.Type, data + (nint)((first + start + i) * rowBytes), row, 0, input);
                    result[start + i] = TensorPrimitives.Norm(row);
                }
            });
            return result;
        }
        finally { if (owned != IntPtr.Zero) QuantizedWeight.FreeBuffer(owned); }
    }

    /// <summary>A PDD bundle's replaced parameters (norm gains, per-step heads) or a 1-D diff.</summary>
    /// <summary>Which plug-in last changed each 1-D gain, and whether it was a full value.</summary>
    private readonly Dictionary<string, (string File, bool Full)> _gainOwners = new(StringComparer.Ordinal);

    /// <summary>A PDD bundle's replaced parameters (norm gains, per-step heads) or a 1-D diff.
    /// Diffs follow ComfyUI's "diff" patch, weight += strength * diff, applied in plug-in order
    /// on top of whatever an earlier plug-in left; a full value refuses to discard an earlier
    /// plug-in's change rather than letting the last one win silently.</summary>
    private int Replace(SafetensorsFile st, string module, int block, string local,
        Dictionary<QwenImage21LoraPart, string> parts, QwenImage21LoraConfig pdd, string file, GgufFile dit, string prefix,
        float strength)
    {
        bool isDiff = parts.TryGetValue(QwenImage21LoraPart.Diff, out string name);
        if (!isDiff) name = parts[QwenImage21LoraPart.Weight];
        var shape = st.GetInfo(name).Shape;
        if (!isDiff && pdd == null)
            throw new InvalidDataException(
                $"LoRA '{file}' carries a full value for {module} ({name}); only VideoX-Fun PDD bundles (with their pdd_config.json) " +
                "replace parameters, and this file does not declare that format.");
        if (!isDiff && pdd.PddFullParameters.Count > 0 && module != "proj_out" && !pdd.PddFullParameters.Contains(module + ".weight") &&
            !pdd.PddFullParameters.Contains(module))
            throw new InvalidDataException($"LoRA '{file}' replaces {module}, which {pdd.Path} does not list in pdd_full_parameters.");
        if (module == "proj_out")
        {
            if (isDiff || shape.Length != 3 || shape[1] != Channels || shape[2] != Dim)
                throw new InvalidDataException($"LoRA '{file}': proj_out.weight [{string.Join(",", shape)}] is not a stack of [{Channels},{Dim}] step heads.");
            if (shape[0] != pdd.PddSteps)
                throw new InvalidDataException($"LoRA '{file}' has {shape[0]} step heads but {pdd.Path} trains {pdd.PddSteps} steps.");
            if (OutputHeads.Length > 0)
                throw new InvalidDataException($"LoRA '{file}' brings per-step output heads, but an earlier plug-in already replaced proj_out.");
            float[] heads = st.ReadFloat32(name);
            var result = new IntPtr[shape[0]];
            for (int h = 0; h < result.Length; h++)
                result[h] = Copy(heads.AsSpan(h * Channels * Dim, Channels * Dim));
            OutputHeads = result;
            return 1;
        }
        long length = local switch { "txt_in.text_norm" => QwenImage21DiT.TextDim, "attn.norm_q" or "attn.norm_k" => QwenImage21DiT.HeadDim, _ => -1 };
        if (length < 0 || !(shape.Length == 1 && shape[0] == length))
            throw new InvalidDataException($"LoRA '{file}': {name} has shape [{string.Join(",", shape)}], which does not replace {module}.");
        IntPtr current = local == "txt_in.text_norm" ? TextNorm : local == "attn.norm_q" ? NormQ[block] : NormK[block];
        if (!isDiff && _gainOwners.TryGetValue(module, out var owner))
            throw new InvalidDataException(
                $"LoRA '{file}' replaces {module}, which '{owner.File}' already changed; the replacement would discard that change. " +
                "List the bundle first, or drop one of the two.");
        float[] value = st.ReadFloat32(name);
        if (isDiff)
        {
            float[] basis;
            if (current != IntPtr.Zero)
                unsafe { basis = new ReadOnlySpan<float>((void*)current, (int)length).ToArray(); }
            else
            {
                var info = dit.Tensors[prefix + module + ".weight"];
                basis = new float[length];
                IntPtr src = QuantizedWeight.AllocateBuffer(dit.GetTensorByteCount(info));
                try
                {
                    dit.ReadTensorDataToNative(info, src, dit.GetTensorByteCount(info));
                    NativeDequant.DequantizeToFloat32((int)info.Type, src, basis, 0, length);
                }
                finally { QuantizedWeight.FreeBuffer(src); }
            }
            TensorPrimitives.MultiplyAdd(value, strength, basis, value);
        }
        IntPtr copy = Copy(value);
        if (local == "txt_in.text_norm") TextNorm = copy;
        else if (local == "attn.norm_q") NormQ[block] = copy;
        else NormK[block] = copy;
        _gainOwners[module] = (file, !isDiff);
        return 1;
    }

    private unsafe IntPtr Copy(ReadOnlySpan<float> values)
    {
        IntPtr p = Allocate(values.Length * sizeof(float));
        values.CopyTo(new Span<float>((void*)p, values.Length));
        return p;
    }

    private unsafe IntPtr Allocate(long bytes)
    {
        void* p = NativeMemory.AlignedAlloc((nuint)Math.Max(bytes, 1), (nuint)PageAlignment);
        if (p == null) throw new OutOfMemoryException($"Unable to allocate {bytes} bytes for LoRA factors.");
        var ptr = (IntPtr)p;
        lock (_buffers) _buffers.Add(ptr);
        return ptr;
    }

    // ---- packing ------------------------------------------------------------------------

    /// <summary>A packed update: native pointers plus what TP sharding needs.</summary>
    private sealed class Packed
    {
        internal QwenImage21Lora Lora;
        internal int Padded;            // rank after padding (== Lora.Rank)
        internal int ElementBytes;      // 2 (F16) or 4 (F32)
    }

    private readonly Dictionary<string, Packed> _packed = new(StringComparer.Ordinal);
    private int _ranks = 1;

    private static int Pad(int rank, int multiple) => rank == 0 ? 0 : (rank + multiple - 1) / multiple * multiple;

    private void Pack(GgufFile dit, string prefix, int multiple, int ranks)
    {
        _ranks = ranks;
        if (OutputHeads.Length > 0 && _slots.TryGetValue("proj_out", out var projOut) && !projOut.Empty)
            throw new InvalidDataException(
                "A VideoX-Fun PDD bundle replaces proj_out with its per-step output heads, so another plug-in's proj_out update " +
                "(low-rank factors or a DoRA magnitude) cannot apply. Remove proj_out from that LoRA or load it without the bundle.");
        bool fused = dit.Tensors.ContainsKey(prefix + "transformer_blocks.0.img_mlp.gate_up.weight");
        // The groups are independent (each owns its allocations), so they pack in parallel.
        var groups = new List<(Func<(string Name, Slot Slot)[]> Group, bool ShareDown)>();
        // Globals: independent inputs, one update each.
        foreach (var module in QwenImage21LoraKeys.GlobalModules)
            if (_slots.TryGetValue(module, out var slot) && !slot.Empty)
                groups.Add((() => new[] { (module, slot) }, false));
        for (int b = 0; b < QwenImage21DiT.Layers; b++)
        {
            Slot Get(string n) => _slots.TryGetValue($"{b}.{n}", out var s) && !s.Empty ? s : null;
            var qkv = new[] { "q", "k", "v" }.Select(n => ($"{b}.{n}", Get(n))).Where(p => p.Item2 != null).ToArray();
            if (qkv.Length > 0) groups.Add((() => qkv, false));
            foreach (var n in new[] { "out", "down" })
                if (Get(n) is { } s)
                {
                    string name = $"{b}.{n}";
                    groups.Add((() => new[] { (name, s) }, false));
                }

            Slot gate = Get("gate"), up = Get("up");
            if (gate == null && up == null) continue;
            // Both halves fed only by fused LoRAs share every down factor.
            bool shared = gate != null && up != null && gate.Terms.Count == up.Terms.Count &&
                gate.Terms.Zip(up.Terms).All(p => ReferenceEquals(p.First.Down, p.Second.Down));
            if (fused && shared && ranks == 1)
            {
                // Keep the checkpoint's fused projection: one update over both halves.
                string name = $"{b}.gate";
                groups.Add((() =>
                {
                    var merged = new Slot { In = gate.In, Out = gate.Out + up.Out };
                    if (gate.RowScale != null || up.RowScale != null)
                        merged.RowScale = (gate.RowScale ?? Enumerable.Repeat(1f, (int)gate.Out).ToArray())
                            .Concat(up.RowScale ?? Enumerable.Repeat(1f, (int)up.Out).ToArray()).ToArray();
                    for (int i = 0; i < gate.Terms.Count; i++)
                        merged.Terms.Add(new Term
                        {
                            Down = gate.Terms[i].Down, Rank = gate.Terms[i].Rank, In = gate.In, Out = merged.Out,
                            Up = gate.Terms[i].Up.Concat(up.Terms[i].Up).ToArray(),
                        });
                    return new[] { (name, merged) };
                }, false));
            }
            else
            {
                if (fused) SplitGateUp[b] = true;
                var halves = new[] { ($"{b}.gate", gate), ($"{b}.up", up) }.Where(p => p.Item2 != null).ToArray();
                groups.Add((() => halves, shared));
            }
        }
        Parallel.ForEach(groups, g => PackGroup(g.Group(), multiple, g.ShareDown));
        BuildAdapters();
        // The F32 terms were only needed to build the native factors.
        _slots.Clear();
    }

    /// <summary>
    /// Pack slots that read one input: their down factors are laid out back to back in one
    /// allocation (the graph's stacked shrink), or, with <paramref name="shareDown"/>, stored
    /// once for all of them.
    /// </summary>
    private unsafe void PackGroup((string Name, Slot Slot)[] group, int multiple, bool shareDown)
    {
        // Rank-concatenated factors per slot, in F32.
        var downs = new List<float[]>();
        var ups = new List<float[]>();
        var paddedRanks = new List<int>();
        foreach (var (_, slot) in group)
        {
            int rank = slot.Terms.Sum(t => t.Rank), padded = Pad(rank, multiple);
            var down = new float[padded * slot.In];
            var up = new float[slot.Out * padded];
            int offset = 0;
            foreach (var t in slot.Terms)
            {
                t.Down.AsSpan().CopyTo(down.AsSpan((int)(offset * slot.In)));
                for (long o = 0; o < slot.Out; o++)
                    t.Up.AsSpan((int)(o * t.Rank), t.Rank).CopyTo(up.AsSpan((int)(o * padded + offset)));
                offset += t.Rank;
            }
            downs.Add(down);
            ups.Add(up);
            paddedRanks.Add(padded);
        }
        // F16 unless some value would overflow it (then the whole group stays F32).
        bool f16 = downs.Concat(ups).All(a => a.Length == 0 || TensorPrimitives.MaxMagnitude(a) < 60000f);
        int elem = f16 ? 2 : 4;
        long totalDown = shareDown ? downs[0].Length : downs.Sum(d => (long)d.Length);
        IntPtr downBase = totalDown > 0 ? Allocate(totalDown * elem) : IntPtr.Zero;
        long cursor = 0;
        for (int i = 0; i < group.Length; i++)
        {
            var (name, slot) = group[i];
            int padded = paddedRanks[i];
            IntPtr down = IntPtr.Zero, up = IntPtr.Zero;
            if (padded > 0)
            {
                down = downBase + (nint)(cursor * elem);
                if (!shareDown || i == 0)
                {
                    Store(downs[i], down, f16);
                    cursor += downs[i].Length;
                }
                else down = downBase;
                up = Allocate(ups[i].LongLength * elem);
                Store(ups[i], up, f16);
                Interlocked.Add(ref _factorBytes, (shareDown && i > 0 ? 0 : downs[i].LongLength * elem) + ups[i].LongLength * elem);
            }
            IntPtr rowScale = slot.RowScale != null ? Copy(slot.RowScale) : IntPtr.Zero;
            var packed = new Packed
            {
                Lora = new QwenImage21Lora
                {
                    Down = down, Up = up, RowScale = rowScale, Type = f16 ? GgmlF16 : GgmlF32,
                    Rank = padded, In = slot.In, Out = slot.Out,
                },
                Padded = padded,
                ElementBytes = elem,
            };
            lock (_packed) _packed[name] = packed;
        }
    }

    private static unsafe void Store(float[] values, IntPtr destination, bool f16)
    {
        if (f16) TensorPrimitives.ConvertToHalf(values, new Span<System.Half>((void*)destination, values.Length));
        else values.CopyTo(new Span<float>((void*)destination, values.Length));
    }

    // ---- native descriptors ---------------------------------------------------------------

    private unsafe void BuildAdapters()
    {
        _adapters = new IntPtr[_ranks];
        for (int r = 0; r < _ranks; r++)
        {
            var blocks = (QwenImage21BlockLora*)Allocate(sizeof(QwenImage21BlockLora) * QwenImage21DiT.Layers);
            for (int b = 0; b < QwenImage21DiT.Layers; b++)
            {
                blocks[b] = new QwenImage21BlockLora
                {
                    Q = Column($"{b}.q", r), K = Column($"{b}.k", r), V = Column($"{b}.v", r),
                    Out = Row($"{b}.out", r),
                    Gate = Column($"{b}.gate", r), Up = Column($"{b}.up", r),
                    Down = Row($"{b}.down", r),
                };
            }
            var adapter = (QwenImage21Adapter*)Allocate(sizeof(QwenImage21Adapter));
            *adapter = new QwenImage21Adapter
            {
                StructBytes = sizeof(QwenImage21Adapter),
                NumLayers = QwenImage21DiT.Layers,
                ImageIn = Global("img_in"), TextIn = Global("txt_in.in_layer"), TextOut = Global("txt_in.out_layer"),
                TimeIn = Global("time_text_embed.timestep_embedder.linear_1"),
                TimeOut = Global("time_text_embed.timestep_embedder.linear_2"),
                Modulation = Global("modulation.1"), NormOut = Global("norm_out.linear"), ProjOut = Global("proj_out"),
                Blocks = (IntPtr)blocks,
                OutputHead = OutputHeads.Length > 0 ? OutputHeads[0] : IntPtr.Zero,
                OutputHeadType = GgmlF32,
            };
            _adapters[r] = (IntPtr)adapter;
        }
    }

    private QwenImage21Lora Global(string module) => _packed.TryGetValue(module, out var p) ? p.Lora : default;

    /// <summary>Column-parallel projection (outputs split): rows of up and of the row scale.</summary>
    private QwenImage21Lora Column(string name, int rank)
    {
        if (!_packed.TryGetValue(name, out var p)) return default;
        if (_ranks == 1) return p.Lora;
        var l = p.Lora;
        long local = l.Out / _ranks;
        return l with
        {
            Up = l.Up == IntPtr.Zero ? IntPtr.Zero : l.Up + (nint)(rank * local * p.Padded * p.ElementBytes),
            RowScale = l.RowScale == IntPtr.Zero ? IntPtr.Zero : l.RowScale + (nint)(rank * local * sizeof(float)),
            Out = local,
        };
    }

    /// <summary>Row-parallel projection (inputs split): columns of down, copied per rank.
    /// Each rank adds up * (down_r * x_r) before the all-reduce, which sums to the update.</summary>
    private unsafe QwenImage21Lora Row(string name, int rank)
    {
        if (!_packed.TryGetValue(name, out var p)) return default;
        if (_ranks == 1) return p.Lora;
        if (p.Lora.Down == IntPtr.Zero) return p.Lora with { In = p.Lora.In / _ranks };
        var l = p.Lora;
        long local = l.In / _ranks;
        IntPtr down = Allocate(p.Padded * local * p.ElementBytes);
        for (long r = 0; r < p.Padded; r++)
            Buffer.MemoryCopy((byte*)l.Down + (r * l.In + rank * local) * p.ElementBytes,
                (byte*)down + r * local * p.ElementBytes, local * p.ElementBytes, local * p.ElementBytes);
        return l with { Down = down, In = local };
    }

    /// <summary>The native adapter descriptor for tensor-parallel rank <paramref name="rank"/>.</summary>
    internal IntPtr AdapterFor(int rank) => _adapters[rank];

    /// <summary>Select the per-step output head (a PDD bundle); a no-op without heads.</summary>
    internal unsafe void SelectOutputHead(int step)
    {
        if (OutputHeads.Length == 0) return;
        if (step < 0 || step >= OutputHeads.Length)
            throw new ArgumentOutOfRangeException(nameof(step), $"The LoRA bundle has {OutputHeads.Length} step heads; step {step + 1} has none.");
        foreach (var adapter in _adapters) ((QwenImage21Adapter*)adapter)->OutputHead = OutputHeads[step];
    }

    public void Dispose()
    {
        foreach (var p in _buffers) unsafe { NativeMemory.AlignedFree(p.ToPointer()); }
        _buffers.Clear();
        _adapters = Array.Empty<IntPtr>();
    }
}
