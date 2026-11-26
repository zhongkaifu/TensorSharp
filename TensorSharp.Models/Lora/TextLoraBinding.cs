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
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace TensorSharp.Models.Lora
{
    /// <summary>One adapter target the loaded model cannot apply, and why.</summary>
    public sealed record TextLoraRejection(string Target, string Reason);

    /// <summary>What a <see cref="TextLoraBinding"/> plan reads from the loaded model.</summary>
    internal interface ITextLoraModelView
    {
        /// <summary>The <c>[out, in]</c> shape of a loaded 2-D weight, or false when there is
        /// none (or the weight of that name is a norm, a bias or another non-matrix).</summary>
        bool TryGetWeightDims(string weightName, out long inDim, out long outDim);

        /// <summary>Whether the model loaded a weight of that name, of any rank.</summary>
        bool HasWeight(string weightName);

        /// <summary>The fused weight that took <paramref name="member"/>'s output rows at load
        /// (attn_q/k/v into attn_qkv, ffn_gate/up into ffn_gate_up), and where they start.</summary>
        bool TryGetFusedMember(string member, out string fusedName, out long rowOffset, out long rows);

        /// <summary>Null when every forward path of the model applies a delta on
        /// <paramref name="weightName"/> (a weight as loaded, after fusion); otherwise why not.</summary>
        string? WeightRefusal(string weightName);

        /// <summary>The model's own name for an adapter target (a family that renames tensors at load).</summary>
        string NormalizeTarget(string target);
    }

    /// <summary>
    /// Everything one model holds for one text-LoRA adapter: the plan (whether, and to which
    /// weights, the adapter applies, decided once from the loaded weights) and, once the
    /// adapter is bound, the factor tensors the forward reads (materialized once per model,
    /// never per request). <c>ModelBase</c> builds it in <c>PlanTextLora</c> and
    /// <c>TryBindAdapter</c>; the request path reads <see cref="IsApplicable"/> and
    /// <see cref="RefusalMessage"/> to answer 400 before anything runs.
    ///
    /// <para>
    /// The plan is deny-by-default. The model refuses outright when it runs on more than one
    /// device, on an unsupported backend, for a family whose projections bypass the hook, or
    /// for an adapter converted for another architecture. Otherwise each target must resolve
    /// to a loaded weight of the right shape, directly or as rows of a projection fused at
    /// load, that the family applies a delta to on every path. One target that does not
    /// refuses the whole adapter, so a request never gets a partially adapted answer.
    /// </para>
    ///
    /// <para>
    /// Targets fused into one weight become one composite factor pair, exactly: the members'
    /// <c>A</c> are stacked along the rank (<c>AT [in, sum r]</c>) and each member's <c>B</c>
    /// is placed at its own output rows of a block matrix (<c>BT [sum r, out]</c>, zeros
    /// elsewhere), so <c>x @ AT @ BT</c> is the concatenation of the members' deltas, plus any
    /// delta named on the fused weight itself. No re-quantization, no approximation, and the
    /// hook in <c>LinearForward</c> stays one lookup per weight.
    /// </para>
    /// </summary>
    public sealed class TextLoraBinding : IDisposable
    {
        // ASCII digits only: \d also matches other scripts' digits, which no GGUF name uses.
        private static readonly Regex s_layerName = new(@"^blk\.([0-9]+)\.", RegexOptions.CultureInvariant);

        /// <summary>One target's share of the delta of a weight of shape <c>[WeightOut, In]</c>:
        /// output rows <c>RowOffset .. RowOffset + Rows</c>.</summary>
        private readonly record struct Contribution(string Target, long RowOffset, long Rows, long In, long WeightOut);

        // Weight name -> the targets that land on it, in name order.
        private readonly Dictionary<string, List<Contribution>> _plan;
        private Dictionary<string, Factors>? _factors;
        // Per applied weight, the serial of the last forward that applied it (forward audit).
        private long[] _stamps = Array.Empty<long>();

        private sealed record Factors(Tensor AT, Tensor BT, int Index);

        private TextLoraBinding(TextLoraAdapter adapter, string? modelRefusal,
            IReadOnlyList<string> applied, IReadOnlyList<TextLoraRejection> rejected,
            Dictionary<string, List<Contribution>> plan)
        {
            Adapter = adapter;
            ModelRefusal = modelRefusal;
            AppliedTargets = applied;
            Rejected = rejected;
            _plan = plan;
            AppliedWeights = plan.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
        }

        public TextLoraAdapter Adapter { get; }

        /// <summary>Why the model cannot apply this adapter at all (mode, backend, family or
        /// architecture), or null.</summary>
        public string? ModelRefusal { get; }

        /// <summary>Adapter targets the model applies, in name order.</summary>
        public IReadOnlyList<string> AppliedTargets { get; }

        /// <summary>The model weights that carry a delta (a fused weight once for all of its
        /// members), in name order.</summary>
        public IReadOnlyList<string> AppliedWeights { get; }

        /// <summary>Targets the model cannot apply, in name order.</summary>
        public IReadOnlyList<TextLoraRejection> Rejected { get; }

        /// <summary>True when every target of the adapter is applied.</summary>
        public bool IsApplicable => ModelRefusal == null && Rejected.Count == 0;

        /// <summary>True once the factor tensors exist (the adapter is bound).</summary>
        public bool IsMaterialized => _factors != null;

        /// <summary>Why the adapter is refused, naming the modules, or null when it applies.</summary>
        public string? RefusalMessage
        {
            get
            {
                if (ModelRefusal != null)
                    return $"LoRA adapter '{Adapter.Name}' cannot be applied: {ModelRefusal}.";
                if (Rejected.Count == 0)
                    return null;
                return $"LoRA adapter '{Adapter.Name}' cannot be applied to the loaded model, and is refused rather " +
                       $"than applied in part. Unsupported modules: {DescribeRejections(Rejected)}.";
            }
        }

        /// <summary>
        /// The modules a refused adapter cannot run with, layers collapsed
        /// (<c>blk.{0-27}.attn_output.weight</c>): every target when the model refuses the
        /// adapter as a whole, otherwise the rejected ones. Empty when it applies.
        /// </summary>
        public IReadOnlyList<string> UnsupportedModules
            => ModelRefusal != null
                ? Group(Adapter.Deltas.Keys.OrderBy(k => k, StringComparer.Ordinal).Select(t => (t, string.Empty))).Select(g => g.Name).ToArray()
                : Group(Rejected.Select(r => (r.Target, r.Reason))).Select(g => g.Name).Distinct(StringComparer.Ordinal).ToArray();

        /// <summary>
        /// Decide where <paramref name="adapter"/> applies. Pure: reads only what
        /// <paramref name="model"/> reports about the loaded weights.
        /// </summary>
        internal static TextLoraBinding Plan(TextLoraAdapter adapter, string? modelRefusal, ITextLoraModelView model)
        {
            ArgumentNullException.ThrowIfNull(adapter);
            ArgumentNullException.ThrowIfNull(model);
            var applied = new List<string>();
            var rejected = new List<TextLoraRejection>();
            var plan = new Dictionary<string, List<Contribution>>(StringComparer.Ordinal);
            if (modelRefusal == null)
            {
                foreach (var (target, delta) in adapter.Deltas.OrderBy(d => d.Key, StringComparer.Ordinal))
                {
                    string? reason = Resolve(target, delta, model, out string weight, out Contribution part);
                    if (reason != null)
                    {
                        rejected.Add(new TextLoraRejection(target, reason));
                        continue;
                    }
                    applied.Add(target);
                    if (!plan.TryGetValue(weight, out var parts))
                        plan[weight] = parts = new List<Contribution>();
                    parts.Add(part);
                }
            }
            return new TextLoraBinding(adapter, modelRefusal, applied, rejected, plan);
        }

        private static string? Resolve(string target, LoraDelta delta, ITextLoraModelView model,
            out string weight, out Contribution part)
        {
            weight = string.Empty;
            part = default;
            string name = model.NormalizeTarget(target);
            // Only decoder projections pass through the hook. token_embd is an embedding
            // lookup (a delta on it would land on a tied LM head instead), and vision,
            // audio and draft-model weights belong to other forwards.
            if (!string.Equals(name, "output.weight", StringComparison.Ordinal) && !s_layerName.IsMatch(name))
                return "not a decoder projection (only blk.N.* and output.weight are adapted)";
            if (name.Contains("_exps", StringComparison.Ordinal))
                return "routed expert weights do not pass through the LoRA hook";

            long rowOffset = 0, rows;
            if (model.TryGetWeightDims(name, out long inDim, out long outDim))
            {
                weight = name;
                rows = outDim;
                if (delta.In != inDim || delta.Out != outDim)
                    return $"weight is [{outDim}, {inDim}] but the delta is [{delta.Out}, {delta.In}]";
            }
            else if (model.TryGetFusedMember(name, out string fused, out rowOffset, out rows)
                     && model.TryGetWeightDims(fused, out inDim, out outDim))
            {
                weight = fused;
                if (delta.In != inDim || delta.Out != rows)
                    return $"its rows of {fused} are [{rows}, {inDim}] but the delta is [{delta.Out}, {delta.In}]";
            }
            else
            {
                return model.HasWeight(name)
                    ? "not a 2-D projection weight (norms and biases take no LoRA delta)"
                    : "not a weight of the loaded model";
            }

            string? refusal = model.WeightRefusal(weight);
            if (refusal != null)
                return weight == name ? refusal : $"fused into {weight}, which {refusal}";
            part = new Contribution(target, rowOffset, rows, inDim, outDim);
            return null;
        }

        /// <summary>
        /// Compact list of rejections: targets that differ only in their layer number and
        /// share a reason collapse into one entry, e.g.
        /// <c>blk.{0-27}.attn_output.weight (not a weight of the loaded model)</c>.
        /// </summary>
        internal static string DescribeRejections(IReadOnlyList<TextLoraRejection> rejected)
        {
            const int MaxGroups = 8;
            var groups = Group(rejected.Select(r => (r.Target, r.Reason)));
            var sb = new StringBuilder();
            for (int i = 0; i < groups.Count && i < MaxGroups; i++)
            {
                if (i > 0) sb.Append("; ");
                sb.Append(groups[i].Name).Append(" (").Append(groups[i].Reason).Append(')');
            }
            if (groups.Count > MaxGroups)
                sb.Append("; and ").Append(groups.Count - MaxGroups).Append(" more");
            return sb.ToString();
        }

        /// <summary>
        /// Group names that differ only in their layer and share a reason. The names are the
        /// adapter's own, so a layer number that does not fit an int
        /// (<c>blk.99999999999.attn_output.weight</c>, a target no model has) is listed as
        /// written rather than grouped; describing a refusal never throws.
        /// </summary>
        private static List<(string Name, string Reason)> Group(IEnumerable<(string Target, string Reason)> items)
        {
            var groups = new List<(string Pattern, string Reason, List<int> Layers)>();
            var index = new Dictionary<(string, string), int>();
            foreach (var (target, reason) in items)
            {
                var match = s_layerName.Match(target);
                string pattern = target;
                int layer = -1;
                if (match.Success
                    && int.TryParse(match.Groups[1].Value, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed))
                {
                    layer = parsed;
                    pattern = "blk.{L}." + target.Substring(match.Length);
                }
                var key = (pattern, reason);
                if (!index.TryGetValue(key, out int at))
                {
                    at = groups.Count;
                    index[key] = at;
                    groups.Add((pattern, reason, new List<int>()));
                }
                if (layer >= 0)
                    groups[at].Layers.Add(layer);
            }
            return groups
                .Select(g => (g.Layers.Count > 0 ? g.Pattern.Replace("{L}", "{" + CompactRanges(g.Layers) + "}") : g.Pattern, g.Reason))
                .ToList();
        }

        private static string CompactRanges(List<int> values)
        {
            values.Sort();
            var sb = new StringBuilder();
            for (int i = 0; i < values.Count;)
            {
                int start = values[i], end = start;
                // Not end + 1, which wraps at int.MaxValue (layers are never negative).
                while (i + 1 < values.Count && values[i + 1] - 1 <= end)
                    end = values[++i];
                i++;
                if (sb.Length > 0) sb.Append(',');
                sb.Append(start.ToString(CultureInfo.InvariantCulture));
                if (end > start) sb.Append('-').Append(end.ToString(CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }

        /// <summary>
        /// Create the factor tensors (<c>AT [in, rank]</c>, <c>BT [rank, out]</c>) for every
        /// weight the adapter applies to, once: a weight with one full-size target uses that
        /// target's factors, any other the composite (see the class remarks). Called under the
        /// model's GPU compute lock.
        /// </summary>
        internal void Materialize(Func<float[], long, long, Tensor> createFloat)
        {
            if (_factors != null)
                return;
            if (!IsApplicable)
                throw new InvalidOperationException(RefusalMessage);
            var factors = new Dictionary<string, Factors>(StringComparer.Ordinal);
            try
            {
                foreach (string weight in AppliedWeights)
                {
                    var (at, bt, inDim, rank, outDim) = Compose(_plan[weight]);
                    Tensor atTensor = createFloat(at, inDim, rank);
                    Tensor btTensor;
                    try { btTensor = createFloat(bt, rank, outDim); }
                    catch { atTensor.Dispose(); throw; }
                    factors[weight] = new Factors(atTensor, btTensor, factors.Count);
                }
            }
            catch
            {
                DisposeFactors(factors);
                throw;
            }
            _stamps = new long[factors.Count];
            _factors = factors;
        }

        /// <summary>
        /// The factor pair of one weight: a lone full-size target's own factors, or the exact
        /// composite of several targets (members of a fused projection, and a delta named on
        /// the fused weight itself).
        /// </summary>
        private (float[] AT, float[] BT, long In, int Rank, long Out) Compose(List<Contribution> parts)
        {
            LoraDelta first = Adapter.Deltas[parts[0].Target];
            long inDim = parts[0].In, outDim = parts[0].WeightOut;
            if (parts.Count == 1 && parts[0].RowOffset == 0 && parts[0].Rows == outDim)
                return (first.AT, first.BT, inDim, first.Rank, outDim);

            int rank = parts.Sum(p => Adapter.Deltas[p.Target].Rank);
            var at = new float[inDim * rank];
            var bt = new float[rank * outDim];
            int r0 = 0;
            foreach (var p in parts)
            {
                LoraDelta d = Adapter.Deltas[p.Target];
                // A stacked along the rank: AT[i, r0 + j] = AT_p[i, j].
                for (long i = 0; i < inDim; i++)
                    Array.Copy(d.AT, i * d.Rank, at, i * rank + r0, d.Rank);
                // B placed at the member's rows: BT[r0 + j, off + c] = BT_p[j, c]; zeros elsewhere.
                for (long j = 0; j < d.Rank; j++)
                    Array.Copy(d.BT, j * p.Rows, bt, (r0 + j) * outDim + p.RowOffset, p.Rows);
                r0 += d.Rank;
            }
            return (at, bt, inDim, rank, outDim);
        }

        /// <summary>The factors for <paramref name="weightName"/>, when the adapter applies to it.</summary>
        internal bool TryGetFactors(string weightName, out (Tensor AT, Tensor BT) factors)
        {
            if (_factors != null && _factors.TryGetValue(weightName, out var f))
            {
                factors = (f.AT, f.BT);
                return true;
            }
            factors = default;
            return false;
        }

        /// <summary>
        /// <see cref="TryGetFactors"/> for the forward, recording that forward
        /// <paramref name="forwardSerial"/> applied the weight (see <see cref="FindUnapplied"/>).
        /// </summary>
        internal bool TryApplyFactors(string weightName, long forwardSerial, out (Tensor AT, Tensor BT) factors)
        {
            if (_factors != null && _factors.TryGetValue(weightName, out var f))
            {
                _stamps[f.Index] = forwardSerial;
                factors = (f.AT, f.BT);
                return true;
            }
            factors = default;
            return false;
        }

        /// <summary>
        /// The weights forward <paramref name="forwardSerial"/> did not apply, layers collapsed,
        /// or null when it applied every one. <c>output.weight</c> counts only when the forward
        /// produced logits (a prompt chunk that stops before the LM head never runs it).
        /// </summary>
        internal string? FindUnapplied(long forwardSerial, bool producedLogits)
        {
            if (_factors == null)
                return null;
            List<(string, string)>? missed = null;
            foreach (var (weight, f) in _factors)
            {
                if (_stamps[f.Index] == forwardSerial)
                    continue;
                if (!producedLogits && string.Equals(weight, "output.weight", StringComparison.Ordinal))
                    continue;
                (missed ??= new List<(string, string)>()).Add((weight, string.Empty));
            }
            if (missed == null)
                return null;
            missed.Sort((x, y) => string.CompareOrdinal(x.Item1, y.Item1));
            return string.Join(", ", Group(missed).Select(g => g.Name));
        }

        /// <summary>One line for the server log, printed once per (model, adapter).</summary>
        internal string DescribePlan(string modelName)
        {
            if (!IsApplicable)
                return $"LoRA '{Adapter.Id}' on {modelName}: refused. {RefusalMessage}";
            int composite = _plan.Count(p => p.Value.Count > 1 || !string.Equals(p.Value[0].Target, p.Key, StringComparison.Ordinal));
            return $"LoRA '{Adapter.Id}' on {modelName}: {AppliedTargets.Count} target(s) applied to {AppliedWeights.Count} weight(s)" +
                   (composite > 0 ? $", {composite} of them fused projections" : string.Empty) + $" ({Adapter.Summary}).";
        }

        public void Dispose()
        {
            var factors = _factors;
            _factors = null;
            if (factors != null)
                DisposeFactors(factors);
        }

        private static void DisposeFactors(Dictionary<string, Factors> factors)
        {
            foreach (var f in factors.Values)
            {
                ForgetDeviceCopy(f.AT);
                ForgetDeviceCopy(f.BT);
                f.AT.Dispose();
                f.BT.Dispose();
            }
        }

        /// <summary>A graph that applied the factors bound their host bytes (GGML keeps a device copy keyed by the address): drop
        /// it, so a later tensor at the same address is not read through it. Nothing where the native library is absent.</summary>
        private static void ForgetDeviceCopy(Tensor t)
        {
            if (t.Storage is not (TensorSharp.Cpu.CpuStorage or TensorSharp.GGML.GgmlStorage)) return;
            try
            {
                TensorSharp.GGML.GgmlBasicOps.InvalidateTensorHostBuffer(t);
            }
            catch (DllNotFoundException)
            {
            }
        }
    }
}
