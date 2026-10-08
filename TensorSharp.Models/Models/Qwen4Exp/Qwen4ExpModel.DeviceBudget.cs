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
using System.Text;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// The model side of the VRAM-aware placement: what each GPU has, what each layer
    /// costs, and the span width the plan was made for.
    /// </summary>
    public partial class Qwen4ExpModel
    {
        /// <summary>Widest span when every routed expert stays on the GPUs.</summary>
        internal const int SpanTokensWhenResident = 4096;

        /// <summary>Widest span once any layer routes its experts to the host. Each
        /// such layer streams its whole expert set onto the GPU per prefill span and
        /// pins seam tensors that grow with the width, so a narrower span buys
        /// resident layers - which is what decode speed is made of.</summary>
        internal const int SpanTokensWhenOffloaded = 2048;

        /// <summary>Narrowest span long-context sub-chunking goes to. Streaming the
        /// host-routed experts starts at 128 tokens; below that they run on the host.</summary>
        internal const int MinSpanTokens = 128;

        /// <summary>KV rows a full-width span may read before prefill spans narrow:
        /// QSA's sparse mask holds [rows, tokens] temporaries, so rows x tokens is the
        /// quantity to bound, not tokens alone.</summary>
        internal const int KvRowsAtFullSpan = 16384;

        /// <summary>Activation scratch the vision tower needs on device 0 beyond its
        /// F32 weights (bounded attention tiles over at most ~2.3k patches).</summary>
        internal const long VisionScratchBytes = 512L << 20;

        /// <summary>Bytes the vision tower will hold on device 0 when the host has
        /// said it will load one (weights dequantized to F32, plus scratch).</summary>
        private long _visionReserveBytes;

        /// <summary>Widest span the token graph runs. Longer prefill is cut into
        /// consecutive spans (<see cref="SpanTokensAt"/>). int.MaxValue = no cap,
        /// which is every backend the placement does not plan for.</summary>
        private int _spanTokenCap = int.MaxValue;

        /// <summary>Largest KV-rows x tokens product one span may read.</summary>
        private long _spanKvRowTokenBudget = long.MaxValue;

        /// <summary>TS_Q4E_PREFILL_CHUNK pinned the width; the plan never narrows it.</summary>
        private bool _spanTokenCapPinned;

        /// <summary>Most recent per-device placement, for failure messages.</summary>
        private string _placementSummary;

        /// <summary>Narrows the base prefill warmup to the planned span width when that
        /// is under 2048; a wider span is warmed by <see cref="WarmUpPlannedSpanWidth"/>,
        /// so a reserve that is too small fails at startup rather than on the first
        /// long prompt.</summary>
        protected override int GgmlPrefillChunkWarmupLength
            => _spanTokenCap == int.MaxValue ? 0 : _spanTokenCap;

        internal static int? ParseSpanTokenOverride(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return null;
            if (!int.TryParse(value.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out int tokens)
                || tokens < MinSpanTokens)
            {
                throw new ArgumentException(
                    $"TS_Q4E_PREFILL_CHUNK must be a token count of at least {MinSpanTokens}; got '{value}'.");
            }
            return tokens;
        }

        /// <summary>TS_HOST_MOE_DEVICE_MIN_BATCH exactly as the native seam reads it
        /// (atoi; a negative value keeps the 128 default).</summary>
        internal static int ParseStreamMinTokens(string value)
        {
            if (value == null)
                return 128;
            int i = 0;
            while (i < value.Length && char.IsWhiteSpace(value[i])) i++;
            bool negative = false;
            if (i < value.Length && (value[i] == '+' || value[i] == '-'))
            {
                negative = value[i] == '-';
                i++;
            }
            long n = 0;
            while (i < value.Length && value[i] >= '0' && value[i] <= '9' && n <= int.MaxValue)
            {
                n = n * 10 + (value[i] - '0');
                i++;
            }
            if (negative) n = -n;
            return n >= 0 ? (int)Math.Min(n, int.MaxValue) : 128;
        }

        private static int StreamMinTokens()
            => ParseStreamMinTokens(Environment.GetEnvironmentVariable("TS_HOST_MOE_DEVICE_MIN_BATCH"));

        internal static Qwen4ExpSpanScratch SpanScratchFor(int tokens, int maxContextLength, int streamMinTokens,
            long extraBytesPerKvRowToken = 0)
            => new(tokens, (long)tokens * Math.Min(Math.Max(1, maxContextLength), KvRowsAtFullSpan), streamMinTokens,
                Qwen4ExpSpanScratch.SeamBytesPerTokenPerHostLayer, extraBytesPerKvRowToken);

        /// <summary>The span only runs flash attention over an F16 cache. Over an F32
        /// one, each attention layer materializes F32 scores and probabilities for
        /// every head: [rows, tokens, heads] twice.</summary>
        private long NonFlashBytesPerKvRowToken(int kvElementBytes)
            => kvElementBytes == 4 ? 2L * Config.NumHeads * sizeof(float) : 0;

        private Qwen4ExpSpanScratch SpanScratch(int tokens, int maxContextLength, int kvElementBytes)
            => SpanScratchFor(tokens, maxContextLength, StreamMinTokens(), NonFlashBytesPerKvRowToken(kvElementBytes));

        private void SetSpanTokenCap(int tokens, int maxContextLength)
        {
            _spanTokenCap = tokens;
            _spanKvRowTokenBudget = (long)tokens * Math.Min(Math.Max(1, maxContextLength), KvRowsAtFullSpan);
        }

        /// <summary>Tokens the span starting at <paramref name="startPos"/> may cover:
        /// the plan's width while the KV it reads is short, narrowing past
        /// <see cref="KvRowsAtFullSpan"/> rows so rows x tokens stays within budget.</summary>
        internal static int SpanTokensAt(int startPos, int cap, long kvRowTokenBudget, int maxContextLength)
        {
            if (cap == int.MaxValue)
                return int.MaxValue;
            long rows = Math.Max(1L, Math.Min((long)Math.Max(1, maxContextLength), (long)startPos + cap));
            long width = Math.Min(cap, kvRowTokenBudget / rows);
            // A width is also a graph shape; keep the set of shapes small.
            width -= width % 64;
            return (int)Math.Clamp(width, Math.Min(MinSpanTokens, cap), cap);
        }

        /// <summary>Free and total bytes of one rank's GPU. The native query reports
        /// the calling thread's ACTIVE rank, so this switches to the rank and back.</summary>
        private static bool TryGetRankMemory(int rank, out long free, out long total)
        {
            free = total = 0;
            int previous = GgmlBasicOps.GetActiveRank();
            try
            {
                if (rank != previous)
                    GgmlBasicOps.SetActiveRank(rank);
                return GgmlBasicOps.TryGetDeviceMemoryInfo(out free, out total) && total > 0 && free >= 0;
            }
            catch (ArgumentOutOfRangeException)
            {
                // The rank is not initialized (yet, or any more): nothing to measure.
                return false;
            }
            finally
            {
                if (rank != previous)
                {
                    try
                    {
                        GgmlBasicOps.SetActiveRank(previous);
                        // The CUDA query left the queried GPU current on this thread;
                        // query the restored rank so the thread's CUDA device matches it.
                        GgmlBasicOps.TryGetDeviceMemoryInfo(out _, out _);
                    }
                    catch (ArgumentOutOfRangeException) { }
                }
            }
        }

        /// <summary>Bytes of one KV / QSA / recurrent element as the cache will be
        /// allocated: F16 unless the operator pinned F32 (qwen4exp refuses the block
        /// quantized types and falls back to F16).</summary>
        private static int PlannedKvElementBytes()
            => KvCacheDtypeConfig.IsExplicitlySet && KvCacheDtypeConfig.Current == KvCacheDtype.F32 ? 4 : 2;

        /// <summary>Device bytes a layer's caches and recurrent state take at
        /// <paramref name="capacity"/> KV rows.</summary>
        private long LayerCacheBytes(int layer, int capacity, int kvElementBytes)
        {
            long bytes = 0;
            if (_isRecurrent[layer])
            {
                long convDim = (long)_headKDim * _numKHeads * 2 + (long)_headVDim * _numVHeads;
                bytes += (long)Math.Max(0, _convKernel - 1) * convDim * sizeof(float);
                bytes += (long)_headKDim * _headVDim * _numVHeads * sizeof(float);
            }
            else
            {
                bytes += 2L * Config.NumKVHeads * capacity * Config.HeadDim * kvElementBytes;
                if (UsesQsa(layer))
                    bytes += (long)capacity * _indexerHeadDim * kvElementBytes;
            }
            if (_isPle[layer] && _pleHeads > 0)
                bytes += (long)Math.Max(0, _pleConvKernel - 1) * Math.Max(1, _pleNgram) * _hcDim * sizeof(float);
            return bytes;
        }

        /// <summary>The same per layer, from the caches as they are allocated now.</summary>
        private long AllocatedLayerCacheBytes(int layer, int kvElementBytes)
        {
            if (_isRecurrent[layer] || _kCache == null)
                return LayerCacheBytes(layer, _kvCacheCapacity, kvElementBytes);
            long bytes = 0;
            foreach (Tensor[] caches in new[] { _kCache, _vCache, _idxKCache })
                if (caches != null && caches[layer] != null)
                    bytes += caches[layer].Storage.ByteLength;
            if (_isPle[layer] && _pleHeads > 0)
                bytes += (long)Math.Max(0, _pleConvKernel - 1) * Math.Max(1, _pleNgram) * _hcDim * sizeof(float);
            return bytes;
        }

        /// <summary>
        /// Attribute the device-bound weights to layers, device 0 (embedding and other
        /// shared tensors) or the last device (final mixer, LM head). With
        /// <paramref name="floatsOnly"/> only the F32 weights are counted: after the
        /// preload the quantized ones are already in the measured free memory.
        /// Mirrors the preload's own decisions, including the token embedding it serves
        /// from the host when its type has no CUDA get_rows.
        /// </summary>
        private void AccumulateDeviceWeightBytes(long[] layerBytes, ref long firstDevice, ref long lastDevice,
            bool floatsOnly)
        {
            if (!floatsOnly)
            {
                bool hasOutput = _quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight");
                foreach (var kv in _quantWeights)
                {
                    if (_stackedExpertMemberNames.Contains(kv.Key) || !ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                        continue;
                    if (string.Equals(kv.Key, "token_embd.weight", StringComparison.Ordinal)
                        && hasOutput && !CanUseGgmlQuantizedGetRows(kv.Value.GgmlType))
                        continue;
                    AttributeWeight(kv.Key, kv.Value.RawBytes, layerBytes, ref firstDevice, ref lastDevice);
                }
            }
            foreach (var kv in _weights)
                AttributeWeight(kv.Key, kv.Value.Storage.ByteLength, layerBytes, ref firstDevice, ref lastDevice);
        }

        private void AttributeWeight(string name, long bytes, long[] layerBytes, ref long firstDevice, ref long lastDevice)
        {
            if (IsHeadSpanWeight(name)) { lastDevice += bytes; return; }
            AccumulateWeightBytes(name, bytes, layerBytes, ref firstDevice);
        }

        private long[] LayerExpertByteArray()
        {
            var bytes = new long[Config.NumLayers];
            for (int l = 0; l < bytes.Length; l++) bytes[l] = LayerExpertBytes(l);
            return bytes;
        }

        /// <summary>--n-cpu-moe / --cpu-moe as a host set, or null when unset.</summary>
        private bool[] ExplicitHostLayers()
        {
            if (!MoeCpuOffloadConfig.IsExplicitlySet)
                return null;
            var host = new bool[Config.NumLayers];
            if (MoeCpuOffloadConfig.IsEnabled)
                for (int l = 0; l < host.Length; l++) host[l] = MoeCpuOffloadConfig.IsLayerOnCpu(l);
            return host;
        }

        /// <summary>
        /// The GPU that cannot hold its share of a tensor-parallel placement, or -1.
        /// Every rank holds the replicated bytes and an even share of the routed
        /// experts; rank 0 also holds <paramref name="rank0OnlyBytes"/>. The sum is a
        /// lower bound (no prefill scratch beyond <paramref name="reserveBytes"/>), so a
        /// rank it rejects cannot be made to fit by anything the load does later.
        /// </summary>
        internal static int FindUnfitTensorParallelRank(long replicatedBytes, long expertBytes, long rank0OnlyBytes,
            long reserveBytes, long[] available, out long need)
        {
            int ranks = available.Length;
            long share = (expertBytes + ranks - 1) / ranks;
            int worst = -1;
            long worstDeficit = 0;
            need = 0;
            for (int r = 0; r < ranks; r++)
            {
                long rankNeed = checked(replicatedBytes + share + reserveBytes + (r == 0 ? rank0OnlyBytes : 0));
                long deficit = rankNeed - available[r];
                if (deficit > worstDeficit)
                {
                    worstDeficit = deficit;
                    worst = r;
                    need = rankNeed;
                }
            }
            return worst;
        }

        /// <summary>
        /// Under --tp every routed expert stays on the GPUs - qwen4exp cannot run them
        /// from the host in that mode - split evenly across the ranks, and everything
        /// else (attention, recurrent and PLE weights, the head, every cache) is held by
        /// every GPU. A model that cannot fit that way is refused here, before the
        /// expert slicing copies every expert into per-rank host buffers and before the
        /// preload, naming the mode that does fit: on 2x 20 GB a 62 GiB expert set would
        /// otherwise run the host out of RAM and then the GPUs out of VRAM.
        /// </summary>
        private void RefuseUnfitTensorParallel(int initialCacheLength)
        {
            if (!IsTensorParallel || !IsGgmlBackend)
                return;
            int ranks = TpDegree;
            var free = new long[ranks];
            var available = new long[ranks];
            for (int r = 0; r < ranks; r++)
            {
                if (!TryGetRankMemory(r, out free[r], out long total))
                    return;     // nothing to measure against; warmup reports a real shortfall
                available[r] = Math.Max(0, free[r] - GpuMemoryBudget.ResolveHeadroomBytes(total));
            }

            int n = Config.NumLayers, kvElement = PlannedKvElementBytes();
            var dense = new long[n];
            long first = 0, last = _mtpResidentBytes;
            AccumulateDeviceWeightBytes(dense, ref first, ref last, floatsOnly: false);
            long weights = first + last, caches = 0, experts = 0;
            for (int l = 0; l < n; l++)
            {
                weights += dense[l];
                caches += LayerCacheBytes(l, initialCacheLength, kvElement);
                experts += LayerExpertBytes(l);
            }
            // The shared experts are sliced across the ranks too.
            foreach (var kv in _quantWeights)
                if (kv.Key.EndsWith("_shexp.weight", StringComparison.Ordinal) && ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                {
                    weights -= kv.Value.RawBytes;
                    experts += kv.Value.RawBytes;
                }
            int rank = FindUnfitTensorParallelRank(weights + caches, experts, _visionReserveBytes,
                Qwen4ExpSpanScratch.FixedBytes, available, out long need);
            if (rank < 0)
                return;

            long share = (experts + ranks - 1) / ranks;
            throw new ModelLoadRefusedException(
                $"qwen4exp: --tp {ranks} cannot hold this model. Under tensor parallelism every routed expert stays "
                + $"on the GPUs ({GiB(experts)} of them, {GiB(share)} per GPU) and every GPU also holds its own copy "
                + $"of the other weights ({GiB(weights)}) and of the caches ({GiB(caches)} at {initialCacheLength} "
                + $"tokens), so each GPU needs at least {GiB(need)}; GPU {rank} has {GiB(available[rank])} after "
                + $"headroom ({GiB(free[rank])} free). Use --layer-split {ranks} instead: it keeps what fits on the "
                + "GPUs and runs the remaining experts from system RAM. Or use GPUs with that much memory free each.");
        }

        /// <summary>
        /// Phase one of a ggml_cuda layer split, before anything uploads: measure every
        /// GPU, price every layer, and choose the runs and the host set together. The
        /// runs are final once the preload starts; the host set is re-fitted against
        /// measured memory after the preload (<see cref="PlanCudaSplitExpertPlacement"/>).
        /// Returns false when the GPUs cannot be measured, leaving the byte balance.
        /// </summary>
        private bool TryPlanCudaLayerSplit(int maxContextLength, int initialCacheLength, int[] fixedMap)
        {
            int devices = LayerSplitDegree, n = Config.NumLayers;
            var free = new long[devices];
            var total = new long[devices];
            var capacity = new long[devices];
            for (int d = 0; d < devices; d++)
            {
                if (!TryGetRankMemory(d, out free[d], out total[d]))
                    return false;
                capacity[d] = Math.Max(0, free[d] - GpuMemoryBudget.ResolveHeadroomBytes(total[d]));
            }

            int kvElement = PlannedKvElementBytes();
            var dense = new long[n];
            long first = _visionReserveBytes, last = _mtpResidentBytes;
            AccumulateDeviceWeightBytes(dense, ref first, ref last, floatsOnly: false);
            for (int l = 0; l < n; l++)
                dense[l] += LayerCacheBytes(l, initialCacheLength, kvElement);
            long[] experts = LayerExpertByteArray();
            bool[] forced = ExplicitHostLayers();
            int? pinned = ParseSpanTokenOverride(Environment.GetEnvironmentVariable("TS_Q4E_PREFILL_CHUNK"));
            int streamMin = StreamMinTokens();
            long nonFlash = NonFlashBytesPerKvRowToken(kvElement);

            int tokens = pinned ?? SpanTokensWhenResident;
            var plan = PlanQwen4ExpPlacement(dense, experts, capacity, first, last,
                SpanScratchFor(tokens, maxContextLength, streamMin, nonFlash), fixedMap, forced);
            if (pinned == null && plan.HostLayers > 0)
            {
                tokens = SpanTokensWhenOffloaded;
                plan = PlanQwen4ExpPlacement(dense, experts, capacity, first, last,
                    SpanScratchFor(tokens, maxContextLength, streamMin, nonFlash), fixedMap, forced);
            }
            _spanTokenCapPinned = pinned != null;
            SetSpanTokenCap(tokens, maxContextLength);

            if (!plan.Fits)
            {
                throw new ModelLoadRefusedException(DescribeUnfitPlan(plan, dense, experts, free, total, capacity,
                    first, last, SpanScratchFor(tokens, maxContextLength, streamMin, nonFlash), fixedMap, forced, maxContextLength,
                    initialCacheLength));
            }

            _layerDevice = plan.LayerDevice;
            var line = new StringBuilder();
            line.Append(CultureInfo.InvariantCulture, $"  Layer split across {devices} GPUs (sized to free VRAM");
            if (fixedMap != null) line.Append(", runs from TS_Q4E_LAYER_SPLIT");
            if (forced != null) line.Append(", host set from --n-cpu-moe/--cpu-moe");
            line.Append("):");
            for (int d = 0, begin = 0; d < devices; d++)
            {
                int end = begin;
                while (end < n && plan.LayerDevice[end] == d) end++;
                int host = 0;
                for (int l = begin; l < end; l++) if (plan.ExpertOnHost[l]) host++;
                line.Append(CultureInfo.InvariantCulture,
                    $" gpu{d}=layers {begin}-{end - 1} ({end - begin - host} with experts on the GPU, {host} on the host), "
                    + $"{GiB(plan.DeviceBytes[d])} of {GiB(free[d])} free planned;");
                begin = end;
            }
            line.Length--;
            line.Append('.');
            Console.WriteLine(line.ToString());
            return true;
        }

        /// <summary>
        /// Phase two of a ggml_cuda layer split, after the preload and the cache
        /// allocation: re-fit each GPU's host set against what that GPU actually has
        /// free now. The runs cannot move any more (the preload released the host
        /// copies), so a GPU that cannot hold its run even with every expert on the
        /// host refuses the load with the numbers instead of failing in warmup.
        /// </summary>
        private void PlanCudaSplitExpertPlacement()
        {
            int devices = LayerSplitDegree, n = Config.NumLayers;
            var pending = new long[n];
            long first = _visionReserveBytes, last = _mtpReady ? 0 : _mtpResidentBytes;
            AccumulateDeviceWeightBytes(pending, ref first, ref last, floatsOnly: true);
            int kvElement = _kvCacheDtype == KvCacheDtype.F32 ? 4 : 2;
            for (int l = 0; l < n; l++)
                pending[l] += AllocatedLayerCacheBytes(l, kvElement);
            long[] experts = LayerExpertByteArray();
            bool[] forced = ExplicitHostLayers();

            var free = new long[devices];
            var available = new long[devices];
            for (int d = 0; d < devices; d++)
            {
                if (!TryGetRankMemory(d, out free[d], out long total))
                {
                    // Nothing to fit against: keep what was asked for explicitly, and
                    // the span width that goes with it.
                    bool anyForced = false;
                    if (forced != null)
                        for (int l = 0; l < n; l++) anyForced |= _expertOnHost[l] = forced[l];
                    if (!_spanTokenCapPinned && anyForced && _spanTokenCap > SpanTokensWhenOffloaded)
                        SetSpanTokenCap(SpanTokensWhenOffloaded, _maxContextLength);
                    return;
                }
                available[d] = Math.Max(0, free[d] - GpuMemoryBudget.ResolveHeadroomBytes(total));
            }

            int tokens = _spanTokenCap == int.MaxValue ? SpanTokensWhenResident : _spanTokenCap;
            int streamMin = StreamMinTokens();
            var host = new bool[n];
            var resident = new int[devices];
            var used = new long[devices];
            for (int attempt = 0; attempt < 2; attempt++)
            {
                var scratch = SpanScratchFor(tokens, _maxContextLength, streamMin, NonFlashBytesPerKvRowToken(kvElement));
                bool anyHost = false;
                for (int d = 0, begin = 0; d < devices; d++)
                {
                    int end = begin;
                    while (end < n && _layerDevice[end] == d) end++;
                    long fixedBytes = (d == 0 ? first : 0) + (d == devices - 1 ? last : 0);
                    int r = FitResidentLayers(pending, experts, begin, end, available[d], fixedBytes, scratch,
                        forced, out long bytes);
                    if (r < 0)
                    {
                        // An explicit host set may just be too small; say so when routing
                        // the rest of this GPU's layers to the host would have fitted.
                        if (forced != null && FitResidentLayers(pending, experts, begin, end, available[d],
                                fixedBytes, scratch, null) >= 0)
                        {
                            throw new ModelLoadRefusedException(
                                $"qwen4exp: the --n-cpu-moe / --cpu-moe placement leaves GPU {d} (layers {begin}-{end - 1}) "
                                + $"needing {GiB(bytes)} with {GiB(available[d])} available after headroom "
                                + $"({GiB(free[d])} free). Omit --n-cpu-moe to let TensorSharp place the experts per GPU, "
                                + "or raise N.");
                        }
                        throw new ModelLoadRefusedException(
                            $"qwen4exp: GPU {d} cannot hold layers {begin}-{end - 1} even with every routed expert "
                            + $"on the host: it needs {GiB(bytes)} for their dense weights, KV cache "
                            + $"({_kvCacheCapacity} rows) and {tokens}-token span scratch, and has {GiB(available[d])} "
                            + $"after headroom ({GiB(free[d])} free). Lower MAX_CONTEXT or TS_Q4E_PREFILL_CHUNK, "
                            + "free GPU memory, or add GPUs.");
                    }
                    resident[d] = r;
                    used[d] = bytes;
                    for (int l = begin; l < end; l++)
                    {
                        host[l] = forced != null ? forced[l] : l < end - r;
                        anyHost |= host[l];
                    }
                    begin = end;
                }
                if (!anyHost || _spanTokenCapPinned || tokens <= SpanTokensWhenOffloaded)
                    break;
                tokens = SpanTokensWhenOffloaded;
            }
            SetSpanTokenCap(tokens, _maxContextLength);
            for (int l = 0; l < n; l++) _expertOnHost[l] = host[l];

            var summary = new StringBuilder();
            long hostBytes = 0;
            int hostLayers = 0;
            for (int l = 0; l < n; l++)
                if (host[l]) { hostLayers++; hostBytes += experts[l]; }
            for (int d = 0, begin = 0; d < devices; d++)
            {
                int end = begin;
                while (end < n && _layerDevice[end] == d) end++;
                long deviceExperts = 0;
                for (int l = begin; l < end; l++) if (!host[l]) deviceExperts += experts[l];
                if (summary.Length > 0) summary.Append("; ");
                summary.Append(CultureInfo.InvariantCulture,
                    $"gpu{d} holds the experts of {resident[d]} of layers {begin}-{end - 1} ({GiB(deviceExperts)}), "
                    + $"{GiB(used[d])} of {GiB(available[d])} after headroom");
                begin = end;
            }
            _placementSummary = summary.ToString();
            string how = forced != null ? "--n-cpu-moe / --cpu-moe" : "planned per GPU";
            Console.WriteLine(hostLayers == 0
                ? $"[moe-offload] qwen4exp ({how}): every layer's routed experts stay on the GPUs; {_placementSummary}."
                : $"[moe-offload] qwen4exp ({how}): routed experts of {hostLayers} of {n} layers run on the host "
                  + $"from the GGUF mapping ({GiB(hostBytes)} read on demand); {_placementSummary}. "
                  + $"Prefill runs in spans of at most {tokens} tokens"
                  + (forced != null ? "." : "; --n-cpu-moe N, TS_Q4E_LAYER_SPLIT and TS_Q4E_PREFILL_CHUNK override."));

            CapContextToDeviceRoom(available, used, kvElement);
        }

        /// <summary>
        /// KV grows on demand past its initial allocation, on each attention layer's
        /// own GPU. After a tight plan that growth has nowhere to go, and a GPU that
        /// runs out mid-conversation fails the span with the recurrent state already
        /// advanced. Cap the context at what the leftover room can actually hold.
        /// </summary>
        private void CapContextToDeviceRoom(long[] available, long[] used, int kvElementBytes)
        {
            if (_kvCacheCapacity >= _maxContextLength)
                return;
            int devices = available.Length, n = Config.NumLayers;
            long extraRows = long.MaxValue;
            int tightest = -1;
            for (int d = 0, begin = 0; d < devices; d++)
            {
                int end = begin;
                while (end < n && _layerDevice[end] == d) end++;
                // Only what grows with the rows: K, V and the QSA key cache. (The PLE
                // history and recurrent state are fixed and already in `used`.)
                long perRow = 0;
                for (int l = begin; l < end; l++)
                    if (!_isRecurrent[l])
                        perRow += LayerCacheBytes(l, 1, kvElementBytes) - LayerCacheBytes(l, 0, kvElementBytes);
                begin = end;
                if (perRow == 0) continue;
                long rows = Math.Max(0, available[d] - used[d]) / perRow;
                if (rows < extraRows) { extraRows = rows; tightest = d; }
            }
            if (tightest < 0)
                return;
            long fit = (long)_kvCacheCapacity + extraRows;
            fit -= fit % 256;
            if (fit >= _maxContextLength)
                return;
            int capped = (int)Math.Max(_kvCacheCapacity, fit);
            Console.WriteLine(
                $"[moe-offload] qwen4exp: context capped at {capped} tokens (was {_maxContextLength}): GPU {tightest} "
                + "has no room to grow the KV cache further after this placement. Set MAX_CONTEXT to reserve a larger "
                + "cache up front, or move more experts to the host with --n-cpu-moe N.");
            _maxContextLength = capped;
        }

        /// <summary>
        /// The context cap for a single-GPU placement the planner did not choose (an
        /// explicit --n-cpu-moe / --cpu-moe): price what that placement holds and cap
        /// the context at what the rest of the card can grow the KV cache into.
        /// </summary>
        private void CapSingleDeviceContext()
        {
            if (_expertOnHost == null || !GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || total <= 0)
                return;
            int n = Config.NumLayers;
            int kvElement = _kvCacheDtype == KvCacheDtype.F32 ? 4 : 2;
            long used = checked((_mtpReady ? 0 : _mtpResidentBytes) + _visionReserveBytes);
            for (int l = 0; l < n; l++) used = checked(used + AllocatedLayerCacheBytes(l, kvElement));
            foreach (var weight in _weights.Values) used = checked(used + weight.Storage.ByteLength);
            int hostLayers = 0;
            long largestHost = 0;
            for (int l = 0; l < n; l++)
            {
                long experts = LayerExpertBytes(l);
                if (_expertOnHost[l]) { hostLayers++; largestHost = Math.Max(largestHost, experts); }
                else used = checked(used + experts);
            }
            int tokens = _spanTokenCap == int.MaxValue ? SpanTokensWhenResident : _spanTokenCap;
            used = checked(used + SpanScratch(tokens, _maxContextLength, kvElement).Bytes(hostLayers, largestHost));
            CapContextToDeviceRoom(new[] { Math.Max(0, free - GpuMemoryBudget.ResolveHeadroomBytes(total)) },
                new[] { used }, kvElement);
        }

        /// <summary>Bytes the vision tower will need on device 0: every tensor of the
        /// projector dequantized to F32 (Qwen35VisionEncoder keeps them that way) plus
        /// activation scratch. 0 without a projector.</summary>
        internal static long EstimateProjectorDeviceBytes(string projectorPath)
        {
            if (string.IsNullOrWhiteSpace(projectorPath) || !System.IO.File.Exists(projectorPath))
                return 0;
            try
            {
                using var gguf = GgufFile.OpenWithoutSiblingShards(projectorPath);
                long elements = 0;
                foreach (var info in gguf.Tensors.Values)
                    elements = checked(elements + info.NumElements);
                return checked(elements * sizeof(float) + VisionScratchBytes);
            }
            catch (Exception ex) when (ex is System.IO.IOException or InvalidOperationException
                or System.IO.InvalidDataException or NotSupportedException or OverflowException)
            {
                // A projector that cannot be read here fails loudly where it is
                // actually loaded; the plan just cannot price it.
                return 0;
            }
        }

        private string DescribeUnfitPlan(Qwen4ExpPlacementPlan plan, long[] dense, long[] experts,
            long[] free, long[] total, long[] capacity, long first, long last, Qwen4ExpSpanScratch scratch,
            int[] fixedMap, bool[] forced, int maxContextLength, int initialCacheLength)
        {
            int devices = capacity.Length, n = dense.Length;
            var text = new StringBuilder();
            if (forced != null)
            {
                // Say which explicit offload WOULD fit, the way DSV4's planner does.
                int suggestion = -1;
                if (fixedMap == null)
                {
                    for (int count = 0; count <= n && suggestion < 0; count++)
                    {
                        var candidate = new bool[n];
                        for (int l = 0; l < count; l++) candidate[l] = true;
                        if (PlanQwen4ExpPlacement(dense, experts, capacity, first, last, scratch, null, candidate).Fits)
                            suggestion = count;
                    }
                }
                text.Append("qwen4exp: the requested --n-cpu-moe / --cpu-moe placement does not fit these GPUs. ");
                if (suggestion >= 0)
                    text.Append(CultureInfo.InvariantCulture,
                        $"Re-run with --n-cpu-moe {suggestion}, or omit the flag to let TensorSharp place the experts. ");
            }
            else if (fixedMap != null)
            {
                text.Append("qwen4exp: TS_Q4E_LAYER_SPLIT gives a GPU more layers than it can hold even with every "
                    + "routed expert on the host. Remove TS_Q4E_LAYER_SPLIT to let TensorSharp size the split. ");
            }
            else
            {
                text.Append("qwen4exp does not fit on these GPUs even with every routed expert on the host. ");
            }
            for (int d = 0, begin = 0; d < devices; d++)
            {
                int end = begin;
                while (end < n && plan.LayerDevice[end] == d) end++;
                text.Append(CultureInfo.InvariantCulture,
                    $"GPU {d} (layers {begin}-{end - 1}) needs {GiB(plan.DeviceBytes[d])} and has {GiB(capacity[d])} "
                    + $"after {GiB(free[d] - capacity[d])} headroom ({GiB(free[d])} of {GiB(total[d])} free). ");
                begin = end;
            }
            text.Append(CultureInfo.InvariantCulture,
                $"The plan reserves {scratch.MaxTokens}-token prefill span scratch per GPU and a KV cache of "
                + $"{initialCacheLength} rows (MAX_CONTEXT={maxContextLength})");
            if (_visionReserveBytes > 0)
                text.Append(CultureInfo.InvariantCulture, $", plus {GiB(_visionReserveBytes)} for the vision tower on GPU 0");
            text.Append(". Lower MAX_CONTEXT, set TS_Q4E_PREFILL_CHUNK lower, free GPU memory, or add GPUs.");
            return text.ToString();
        }

        /// <summary>
        /// Warmup is the first forward, so it is where a placement that does not
        /// really fit shows up. Two things must not happen then:
        /// <list type="bullet">
        /// <item>an out-of-memory escaping as a plain exception - the hosts treat that
        /// as a crash and print a stack trace, where a load refusal is one line that
        /// says what to change;</item>
        /// <item>the base class swallowing a failed PREFILL warmup ("models with a
        /// per-op fallback survive one"). qwen4exp's span is required whenever QSA or
        /// a layer split is in play, and a failed span latches it off for good, so the
        /// host would start serving a model that can never answer.</item>
        /// </list>
        /// An out-of-memory under an automatic placement first moves more of that GPU's
        /// experts to the host and warms up again (<see cref="TryRouteMoreExpertsToHost"/>);
        /// only when that cannot help does the load refuse.
        /// </summary>
        public override void WarmUpKernels()
        {
            for (int attempt = 0; ; attempt++)
            {
                Exception error = null;
                string failure = null;
                try
                {
                    base.WarmUpKernels();
                    if (!_tokenGraphUnsupported)
                        WarmUpPlannedSpanWidth();
                }
                catch (Exception ex) when (ex is not ModelLoadRefusedException && IsDeviceAllocationFailure(ex))
                {
                    error = ex;
                    failure = DeepestAllocationMessage(ex);
                }
                if (failure == null && _tokenGraphUnsupported && (HasQsa || LayerSplitDegree > 1 || IsTensorParallel))
                    failure = _tokenGraphDeclineReason ?? "the token span declined";
                if (failure == null)
                    return;
                if (IsDeviceAllocationFailure(failure) && attempt < MaxWarmupReplans
                    && TryRouteMoreExpertsToHost(failure))
                    continue;
                throw new ModelLoadRefusedException(IsDeviceAllocationFailure(failure)
                    ? DescribeWarmupOutOfMemory(failure)
                    : "qwen4exp: kernel warmup disabled the token span this model requires, so it cannot serve "
                      + "requests. " + Truncate(failure, 600), error);
            }
        }

        /// <summary>How many times warmup may move more experts to the host and retry.</summary>
        internal const int MaxWarmupReplans = 4;

        /// <summary>
        /// The base prefill warmup stops at 2048 tokens and at a quarter of the
        /// context, so a plan that reserved a wider span (4096 with every expert
        /// resident) would first build that graph on a user's long prompt - and a
        /// reserve that is short would fail there, where nothing can be re-planned.
        /// Build the widest span a prompt can take here instead. Skipped when the
        /// operator set the warmup (TS_PREFILL_WARMUP=0 / TS_PREFILL_WARMUP_LEN) and on
        /// an integrated GPU, where the base warmup is deliberately short.
        /// </summary>
        private void WarmUpPlannedSpanWidth()
        {
            if (!IsGgmlBackend || IsTensorParallel || _spanTokenCap == int.MaxValue
                || string.Equals(Environment.GetEnvironmentVariable("TS_PREFILL_WARMUP"), "0", StringComparison.Ordinal)
                || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_PREFILL_WARMUP_LEN")))
                return;
            int width = SpanTokensAt(0, _spanTokenCap, _spanKvRowTokenBudget, _maxContextLength);
            int warmed = ResolvePrefillWarmupInputLength(
                Math.Min(ResolvePrefillWarmupTargetLength(_backend, false, false, false, NativeCudaPrefillWarmupLength, null),
                    GgmlPrefillChunkWarmupLength),
                MaxContextLength, 0, explicitLength: false);
            if (width <= warmed || width >= MaxContextLength)
                return;
            // Same query, same fallback as the base warmup's integrated-GPU check.
            bool integrated;
            try { integrated = GgmlBasicOps.IsActiveDeviceIntegrated(); }
            catch { integrated = false; }
            if (integrated)
                return;

            int[] prompt = new int[width];
            Array.Fill(prompt, Config.VocabSize > 1 ? 1 : 0);
            Console.WriteLine($"    Span warmup ({width} tokens, the planned prefill width): starting...");
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            bool unwinding = true;
            _inWarmup = true;
            try
            {
                ForwardRefill(prompt);
                double ms = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                Console.WriteLine($"    Span warmup ({width} tokens): completed in {ms:F1} ms");
                unwinding = false;
            }
            // Out-of-memory goes to the caller, which re-plans or refuses; anything else
            // is what the base warmup also tolerates, unless it took the backend down.
            catch (Exception ex) when (!IsDeviceAllocationFailure(ex) && !TryGetBackendFailure(out _))
            {
                Console.WriteLine($"  Span warmup skipped: {ex.GetType().Name}: {ex.Message}");
                unwinding = false;
            }
            finally
            {
                _inWarmup = false;
                // While an out-of-memory unwinds, a reset that fails as well must not
                // replace it: the caller re-plans from that message and resets again.
                if (!TryGetBackendFailure(out _))
                {
                    try { ResetKVCache(); }
                    catch (Exception) when (unwinding) { }
                }
                ResetForwardTiming();
            }
        }

        /// <summary>
        /// The plan was too tight for the GPU warmup ran out on: something else took
        /// memory after it was measured, or the driver needs more than the model
        /// predicts. Warmup runs before any request, so the placement can still move:
        /// route the experts of more of that GPU's resident layers to the host (its
        /// leading ones, as the plan does), release their device copies and every span
        /// graph, reset the sequence state and warm up again. Explicit --n-cpu-moe /
        /// --cpu-moe placements are the operator's to change and are not second-guessed.
        /// </summary>
        private bool TryRouteMoreExpertsToHost(string failure)
        {
            if (!IsGgmlBackend || _spanTokenCap == int.MaxValue || _expertOnHost == null
                || MoeCpuOffloadConfig.IsExplicitlySet || TryGetBackendFailure(out _))
                return false;
            int device = ParseFailedDevice(failure) ?? (LayerSplitDegree <= 1 ? 0 : -1);
            if (device < 0)
                return false;
            var resident = new List<int>();
            for (int l = 0; l < Config.NumLayers; l++)
                if (DeviceForLayer(l) == device && !_expertOnHost[l])
                    resident.Add(l);
            if (resident.Count == 0)
                return false;
            // A quarter of what the GPU holds, at least two layers: the first host-routed
            // layer of a GPU buys a whole stream buffer back, so one layer may free nothing.
            int move = Math.Min(resident.Count, Math.Max(2, (resident.Count + 3) / 4));
            List<int> moved = resident.GetRange(0, move);
            foreach (int l in moved)
            {
                _expertOnHost[l] = true;
                if (_ffnArgs != null)
                    _ffnArgs[l].CpuMoe = 1;
                foreach (string part in new[] { "gate", "up", "down", "gate_up" })
                    if (_stackedExpertWeights.TryGetValue($"blk.{l}.ffn_{part}_exps.weight", out var experts))
                        GgmlBasicOps.InvalidateHostBuffer(experts.Data);
            }
            if (!_spanTokenCapPinned && _spanTokenCap > SpanTokensWhenOffloaded)
                SetSpanTokenCap(SpanTokensWhenOffloaded, _maxContextLength);
            Console.WriteLine(
                $"[moe-offload] qwen4exp: GPU {device} ran out of memory during warmup; routing the experts of "
                + $"layers {moved[0]}-{moved[^1]} to the host as well ({Truncate(FirstAllocationLine(failure), 160)}) "
                + "and warming up again.");
            try
            {
                GgmlBasicOps.Qwen4ExpResetFfnCache();
                ResetKVCache();
            }
            catch (Exception ex) when (ex is InvalidOperationException or OutOfMemoryException
                or NotSupportedException or System.Runtime.InteropServices.SEHException)
            {
                // The caller refuses the load with the original failure; a reset that
                // cannot complete is no basis for another warmup.
                Console.Error.WriteLine($"[moe-offload] qwen4exp: could not reset after the warmup failure: {ex.Message}");
                return false;
            }
            _tokenGraphUnsupported = false;
            _tokenGraphDeclineReason = null;
            return true;
        }

        internal static int? ParseFailedDevice(string failure)
        {
            if (failure == null)
                return null;
            var match = System.Text.RegularExpressions.Regex.Match(failure, @"on device (\d+) failed");
            return match.Success && int.TryParse(match.Groups[1].Value, NumberStyles.None,
                CultureInfo.InvariantCulture, out int device) ? device : null;
        }

        private static string FirstAllocationLine(string failure)
        {
            if (failure == null)
                return string.Empty;
            int at = failure.IndexOf("allocating ", StringComparison.Ordinal);
            if (at < 0) at = failure.IndexOf("failed to allocate", StringComparison.Ordinal);
            if (at < 0) return failure;
            int end = failure.IndexOf('|', at);
            return (end < 0 ? failure.Substring(at) : failure.Substring(at, end - at)).Trim();
        }

        private static string DeepestAllocationMessage(Exception ex)
        {
            string message = ex.Message;
            for (Exception e = ex.InnerException; e != null; e = e.InnerException)
                if (IsDeviceAllocationFailure(e.Message)) message = e.Message;
            return message;
        }

        /// <summary>The span's native errors carry ggml's allocator lines; these are
        /// the ones that mean a GPU ran out. ggml words host failures the same way
        /// ("failed to allocate CPU buffer", "CUDA_Host"), and a managed
        /// OutOfMemoryException is host memory too: advising more VRAM headroom for
        /// those would move more experts into the RAM that is already short.</summary>
        internal static bool IsDeviceAllocationFailure(string message)
            => message != null
               && (message.Contains("cudaMalloc failed", StringComparison.Ordinal)
                   || message.Contains("cudaErrorMemoryAllocation", StringComparison.Ordinal)
                   || message.Contains("could not persist mutable cache buffer", StringComparison.Ordinal)
                   || message.Contains("token span: failed to allocate graph tensors", StringComparison.Ordinal)
                   // The span's per-sequence device state (GDN, PLE history, QSA keys).
                   || message.Contains("state buffer alloc failed", StringComparison.Ordinal)
                   || message.Contains("ple state alloc failed", StringComparison.Ordinal)
                   || message.Contains("QSA: cache allocation failed", StringComparison.Ordinal)
                   || DeviceBufferAllocationFailure.IsMatch(message));

        private static readonly System.Text.RegularExpressions.Regex DeviceBufferAllocationFailure =
            new(@"failed to allocate (CUDA|Vulkan|Metal)\d* buffer", System.Text.RegularExpressions.RegexOptions.CultureInvariant);

        private static bool IsDeviceAllocationFailure(Exception ex)
        {
            for (Exception e = ex; e != null; e = e.InnerException)
                if (IsDeviceAllocationFailure(e.Message))
                    return true;
            return false;
        }

        private string DescribeWarmupOutOfMemory(string nativeMessage)
        {
            var text = new StringBuilder("qwen4exp ran out of GPU memory during kernel warmup");
            string placement = DescribeCurrentPlacement();
            if (placement != null)
                text.Append(" with this placement: ").Append(placement);
            text.Append(". Something else may be using the GPU (close it). ");
            if (MoeCpuOffloadConfig.IsExplicitlySet && !(MoeCpuOffloadConfig.AllLayers))
            {
                // The operator pinned the host set; headroom does not move it.
                text.Append("Otherwise route more experts to the host: raise --n-cpu-moe N (or use --cpu-moe), "
                    + "or omit it to let TensorSharp place them; a lower MAX_CONTEXT also helps. ");
            }
            else if (_spanTokenCap == int.MaxValue)
            {
                // No placement was planned here (tensor parallelism, or a backend the
                // planner does not size), so the planner's knobs change nothing.
                text.Append("Otherwise lower MAX_CONTEXT, or free GPU memory. ");
            }
            else if (MoeCpuOffloadConfig.IsExplicitlySet)
            {
                text.Append("Every routed expert already runs on the host: lower MAX_CONTEXT or TS_Q4E_PREFILL_CHUNK, "
                    + "or add GPUs. ");
            }
            else
            {
                string headroom = Environment.GetEnvironmentVariable("TS_VRAM_HEADROOM_MB");
                long currentHeadroomMb = long.TryParse(headroom, NumberStyles.None, CultureInfo.InvariantCulture, out long mb)
                    ? mb : 0;
                text.Append(CultureInfo.InvariantCulture,
                    $"Otherwise the plan was too tight for this driver: re-run with "
                    + $"TS_VRAM_HEADROOM_MB={Math.Max(currentHeadroomMb, 1024) + 2048} to keep more free on every GPU "
                    + $"(moving more experts to the host), or lower TS_Q4E_PREFILL_CHUNK or MAX_CONTEXT. ");
            }
            text.Append("Native error: ").Append(Truncate(nativeMessage, 600));
            return text.ToString();
        }

        /// <summary>Which GPU holds how many layers' experts right now (after any
        /// warmup re-plan), or null without a placement to describe.</summary>
        private string DescribeCurrentPlacement()
        {
            if (_expertOnHost == null || _layerDevice == null)
                return _placementSummary;
            int n = Config.NumLayers, devices = Math.Max(1, LayerSplitDegree);
            var parts = new List<string>(devices);
            for (int d = 0, begin = 0; d < devices && begin < n; d++)
            {
                int end = begin, resident = 0;
                while (end < n && _layerDevice[end] == d) { if (!_expertOnHost[end]) resident++; end++; }
                if (end == begin) continue;
                parts.Add(devices > 1
                    ? $"gpu{d} holds the experts of {resident} of layers {begin}-{end - 1}"
                    : $"the GPU holds the experts of {resident} of {n} layers");
                begin = end;
            }
            return parts.Count == 0 ? _placementSummary : string.Join("; ", parts);
        }

        private static string Truncate(string text, int max)
            => text == null || text.Length <= max ? text : text.Substring(0, max) + "...";

        private static string GiB(long bytes) => $"{bytes / (double)(1L << 30):F1} GiB";
    }
}
