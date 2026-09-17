// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>One model under the conformance script, with the prompts it runs.</summary>
internal sealed class ConformanceSubject
{
    public required string Name { get; init; }
    /// <summary>Implements <see cref="IModelArchitecture"/>, <see cref="IBatchedPagedModel"/>,
    /// <see cref="IPrefixCacheModel"/> and <see cref="IPrefixCacheModelDiagnostics"/>.</summary>
    public required IModelArchitecture Model { get; init; }
    /// <summary>The capture point P: what every new chat shares.</summary>
    public required int[] SharedPrefix { get; init; }
    /// <summary>What a new chat adds after P.</summary>
    public required int[] Suffix { get; init; }
    public int DecodeTokens { get; init; } = 8;
    /// <summary>Tokens generated before a holder is donated (its last one not yet forwarded).</summary>
    public int DonateAfter { get; init; } = 3;
    /// <summary>Distance of the in-range rewind (aligned and capped by the family's rules).</summary>
    public int RewindTokens { get; init; } = 4;
    public PrefixCacheMode ExpectedReadiness { get; init; } = PrefixCacheMode.Legacy;
    /// <summary>False for native-slot families, whose slot bytes the managed side does not measure.</summary>
    public bool PayloadBytesKnown { get; init; } = true;
    public Action<string> Log { get; init; } = _ => { };
}

/// <summary>Which steps ran and which did not apply to the family, for the test output.</summary>
internal sealed class ConformanceReport
{
    public List<string> Ran { get; } = new();
    public List<string> NotApplicable { get; } = new();
    public override string ToString() =>
        $"ran [{string.Join(", ", Ran)}]; not applicable [{string.Join(", ", NotApplicable)}]";
}

/// <summary>
/// The contract conformance script (DESIGN §14.2 M2, UCF §3.1): the same steps against every
/// implementer of <see cref="IPrefixCacheModel"/> — the seven oracle fakes and, as ModelFacts, the
/// real families. Every reuse is compared token for token with a cold run of the same tokens on the
/// same model, so a copy that misses state, a donation that loses its holder, a truncation outside
/// the family's rules or a settle that does not flush shows up as a different greedy stream.
///
/// <list type="number">
/// <item>capabilities are well formed and inert (the expected readiness);</item>
/// <item>capture a copy of the active cache, which keeps decoding exactly;</item>
/// <item>clone the capture twice, each continuing exactly like a cold prefill;</item>
/// <item>donate a finished holder, return the donation, donate again, continue exactly;</item>
/// <item>clone a donated (device-authoritative) holder: the settle must flush it;</item>
/// <item>truncate in range (and refuse out of range);</item>
/// <item>convert the primary cache into a payload, continue from it, and use the fresh primary;</item>
/// <item>batched release: at most one decode-graph reset, idempotent for released and unknown keys;</item>
/// <item>export and import where persistable;</item>
/// <item>release everything: no payload keys and no private holders remain.</item>
/// </list>
/// </summary>
internal static class PrefixCacheConformanceScript
{
    internal static ConformanceReport Run(ConformanceSubject s)
    {
        var run = new Runner(s);
        run.Execute();
        s.Log($"[{s.Name}] conformance {run.Report}");
        return run.Report;
    }

    private sealed class Runner
    {
        private readonly ConformanceSubject _s;
        private readonly IModelArchitecture _arch;
        private readonly IBatchedPagedModel _paged;
        private readonly IPrefixCacheModel _pcm;
        private readonly IPrefixCacheModelDiagnostics _diag;
        private readonly HashSet<string> _keys = new(StringComparer.Ordinal);
        private readonly HashSet<string> _requests = new(StringComparer.Ordinal);
        private PrefixCacheCapabilities _caps = null!;
        private ResumabilityRules _rules = null!;
        private int _serial;
        private int[] _a = Array.Empty<int>();
        private List<int> _cold = new();

        internal Runner(ConformanceSubject s)
        {
            _s = s;
            _arch = s.Model;
            _paged = s.Model as IBatchedPagedModel ?? throw new ArgumentException($"{s.Name} is not an IBatchedPagedModel");
            _pcm = s.Model as IPrefixCacheModel ?? throw new ArgumentException($"{s.Name} is not an IPrefixCacheModel");
            _diag = s.Model as IPrefixCacheModelDiagnostics ?? throw new ArgumentException($"{s.Name} is not an IPrefixCacheModelDiagnostics");
            if (s.DonateAfter < 1 || s.DonateAfter >= s.DecodeTokens)
                throw new ArgumentException("DonateAfter must lie in [1, DecodeTokens).");
        }

        internal ConformanceReport Report { get; } = new();

        private bool Holders => _caps.EndState != EndStateSupport.None;
        private bool Copies => _caps.EndState == EndStateSupport.CopyAndDonate;
        private int P => _s.SharedPrefix.Length;

        private string Key() => $"pc:conformance:{++_serial}";

        private string Request(string what)
        {
            string id = $"conformance-{what}-{++_serial}";
            _requests.Add(id);
            return id;
        }

        private void Ran(string step)
        {
            Report.Ran.Add(step);
            _s.Log($"[{_s.Name}] {step}: ok");
        }

        private void NotApplicable(string step, string why) => Report.NotApplicable.Add($"{step} ({why})");

        internal void Execute()
        {
            Capabilities();
            _a = _s.SharedPrefix.Concat(_s.Suffix).ToArray();
            _cold = Cold(_a, _s.DecodeTokens);
            _s.Log($"[{_s.Name}] cold over {_a.Length} tokens: {string.Join(",", _cold)}");

            if (!Holders) EndStateRefusals(); else NotApplicable("end-state refusals", "the family has end states");
            string? capture = CaptureCopy();
            Clone(capture);
            DonateReturnDonate();
            SettleThenClone();
            Truncate(capture);
            ConvertPrimary();
            BatchedRelease();
            ExportImport();
            ReleaseEverything();
        }

        // ------------------------------------------------------------------ steps

        private void Capabilities()
        {
            _caps = _pcm.GetPrefixCacheCapabilities();
            Assert.NotNull(_caps);
            Assert.False(string.IsNullOrEmpty(_caps.NamespaceFingerprint), "NamespaceFingerprint must be non-empty (K1a)");
            Assert.Equal(_arch.KVStateFingerprint, _caps.NamespaceFingerprint);
            Assert.Equal(_s.ExpectedReadiness, _caps.Readiness);
            _rules = new ResumabilityRules(_caps, blockSize: 256, batchedPagedEnabled: false);
            Assert.Equal(Math.Max(1, _arch.KVCacheTruncationGranularity), _caps.TruncationGranularity);
            if (_caps.EndState == EndStateSupport.None)
            {
                Assert.False(_caps.CanCaptureCopy);
                Assert.False(_caps.AdoptPrimaryOnDisplacement);
                Assert.False(_caps.Persistable);
            }
            if (_caps.EndState == EndStateSupport.DonateOnly) Assert.False(_caps.CanCaptureCopy);
            if (_caps.Truncation != TruncationKind.None) Assert.True(_arch.SupportsKVCacheTruncation);
            if (_caps.ReuseAcrossMediaSpan) Assert.True(_arch.SupportsReuseAcrossMediaSpan);
            if (_caps.EndState != EndStateSupport.None) Assert.True(_paged.SupportsPerSequenceFusedForward);
            foreach (ResourceClass cls in Enum.GetValues<ResourceClass>())
                Assert.True(_pcm.QuerySpareBytes(cls) >= -1, $"QuerySpareBytes({cls}) must be -1 (unknown) or a byte count");
            Assert.Empty(_diag.RetainedPayloadKeys);
            Ran("capabilities");
        }

        private void EndStateRefusals()
        {
            string id = Request("refusal");
            string key = Key();
            Assert.False(_pcm.TryCaptureCopy(id, key, out _));
            Assert.False(_pcm.TryCaptureDonate(id, key, 1, out _));
            Assert.False(_pcm.TryConvertPrimary(key, 1, out _));
            Assert.False(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, key, id, 1, 1)));
            Assert.False(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, key, id, 1, 1)));
            Assert.False(_pcm.TryReturnDonation(id, key));
            Assert.False(_pcm.CanMaterialize(key, 1, 1));
            Assert.Equal(default, _pcm.MeasureEndState(key));
            Assert.Equal(default, _pcm.EstimateCloneBytes(key, 1));
            Assert.False(_pcm.TryExport(key, new MemoryStream()));
            long resets = _diag.DecodeGraphResets;
            _pcm.ReleasePayloads(new[] { key }, ReleaseReason.Evicted);
            Assert.Equal(resets, _diag.DecodeGraphResets);
            Assert.Empty(_diag.RetainedPayloadKeys);
            Ran("end-state refusals");
        }

        private string? CaptureCopy()
        {
            if (!_caps.CanCaptureCopy) { NotApplicable("capture", "CanCaptureCopy=false"); return null; }
            string id = Request("capture");
            Assert.True(_paged.BindSequenceCache(id));
            _arch.Forward(_s.SharedPrefix);
            string key = Key();
            Assert.True(_pcm.TryCaptureCopy(id, key, out PayloadFootprint fp), "TryCaptureCopy refused");
            _keys.Add(key);
            Assert.Equal(P, fp.Tokens);
            Assert.True(fp.CapacityTokens >= P, $"capacity {fp.CapacityTokens} below {P} tokens");
            if (_s.PayloadBytesKnown) Assert.True(fp.Bytes.TotalBytes > 0, "a captured end state charges bytes");
            Assert.Contains(key, _diag.RetainedPayloadKeys);
            Assert.Equal(fp.Tokens, _pcm.MeasureEndState(key).Tokens);
            Assert.True(_pcm.CanMaterialize(key, P, P));
            Assert.False(_pcm.CanMaterialize(key, P + 1, P + 1), "a payload is materializable only at its own length");
            // The capture must not disturb the source: it keeps decoding like a cold run.
            List<int> continued = Greedy(_arch.Forward(_s.Suffix), _s.DecodeTokens);
            Assert.Equal(_cold, continued);
            Release(id);
            Ran("capture");
            return key;
        }

        private void Clone(string? capture)
        {
            if (!Copies || capture == null) { NotApplicable("clone", Copies ? "no capture" : "donate-only or no end states"); return; }
            if (_s.PayloadBytesKnown) Assert.True(_pcm.EstimateCloneBytes(capture, P).TotalBytes > 0, "a clone allocates bytes");
            for (int i = 0; i < 2; i++)
            {
                string id = Request("clone");
                Assert.True(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, capture, id, P, P)), "clone refused");
                Assert.True(_paged.HasFusedSequenceCache(id));
                Assert.Contains(capture, _diag.RetainedPayloadKeys);   // a clone leaves the payload cached
                Assert.False(_paged.BindSequenceCache(id), "a cloned holder binds as an existing cache");
                Assert.Equal(_cold, Greedy(_arch.Forward(_s.Suffix), _s.DecodeTokens));
                Release(id);
            }
            Assert.False(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, capture, Request("clone-wrong"), P + 1, P + 1)));
            Ran("clone x2");
        }

        private void DonateReturnDonate()
        {
            if (!Holders) { NotApplicable("donate/return/donate", "no end states"); return; }
            (string key, int length, List<int> generated) = DonatedHolder("donate", _a);
            string first = Request("donee");
            Assert.True(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, key, first, length, length)), "donate refused");
            Assert.DoesNotContain(key, _diag.RetainedPayloadKeys);
            Assert.True(_paged.HasFusedSequenceCache(first));
            Assert.True(_pcm.TryReturnDonation(first, key), "return of an unbound donation refused");
            Assert.Contains(key, _diag.RetainedPayloadKeys);
            Assert.False(_paged.HasFusedSequenceCache(first));
            string second = Request("donee");
            Assert.True(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, key, second, length, length)), "second donate refused");
            _keys.Remove(key);
            Assert.False(_paged.BindSequenceCache(second));
            AssertContinuation(generated, _cold);
            Release(second);
            Ran("donate/return/donate");
        }

        private void SettleThenClone()
        {
            if (!Copies) { NotApplicable("settle then clone", "donate-only or no end states"); return; }
            (string key, int length, List<int> generated) = DonatedHolder("settle", _a);
            string clone = Request("settle-clone");
            Assert.True(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, key, clone, length, length)),
                "clone of a donated holder refused: the settle did not make it host-authoritative");
            Assert.False(_paged.BindSequenceCache(clone));
            AssertContinuation(generated, _cold);
            Release(clone);
            // The settled payload itself still continues exactly.
            string donee = Request("settle-donee");
            Assert.True(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, key, donee, length, length)));
            _keys.Remove(key);
            Assert.False(_paged.BindSequenceCache(donee));
            AssertContinuation(generated, _cold);
            Release(donee);
            Ran("settle then clone");
        }

        private void Truncate(string? capture)
        {
            if (_caps.Truncation == TruncationKind.None)
            {
                if (Holders)
                {
                    (string key, int length, _) = DonatedHolder("no-rewind", _a);
                    Assert.False(_pcm.CanMaterialize(key, length, length - 1), "a family without truncation materializes exact lengths only");
                    MaterializeOp op = Copies ? MaterializeOp.Clone : MaterializeOp.Donate;
                    Assert.False(_pcm.TryMaterialize(new MaterializeRequest(op, key, Request("no-rewind"), length, length - 1)));
                    Ran("truncate refused (Truncation=None)");
                }
                else
                {
                    NotApplicable("truncate", "Truncation=None and no end states");
                }
                return;
            }

            // Out of range first: a wrapped ring cannot be rewound, however short the rewind.
            if (_caps.Truncation == TruncationKind.WithinUnwrappedWindow && Holders
                && _a.Length + _s.DonateAfter - 1 > _caps.TruncationParameter)
            {
                (string wrapped, int length, _) = DonatedHolder("wrapped", _a);
                int target = ResumabilityRules.AlignDown(length - _caps.TruncationGranularity, _caps.TruncationGranularity);
                Assert.False(_pcm.CanMaterialize(wrapped, length, target),
                    $"a wrapped end state ({length} > W={_caps.TruncationParameter}) must refuse a rewind to {target}");
                MaterializeOp op = Copies ? MaterializeOp.Clone : MaterializeOp.Donate;
                Assert.False(_pcm.TryMaterialize(new MaterializeRequest(op, wrapped, Request("wrapped"), length, target)));
                Assert.True(_pcm.CanMaterialize(wrapped, length, length), "the refusal must leave the payload intact");
                if (capture != null && P > _caps.TruncationParameter)
                    Assert.False(_pcm.CanMaterialize(capture, P, ResumabilityRules.AlignDown(P - 1, _caps.TruncationGranularity)));
                Ran("truncate refused (wrapped window)");
            }

            // The rewind sequence: short enough to stay inside an unwrapped window.
            int[] sequence = _a;
            if (_caps.Truncation == TruncationKind.WithinUnwrappedWindow)
            {
                int limit = _caps.TruncationParameter - _s.DonateAfter;
                if (limit < _s.RewindTokens + 2) { NotApplicable("truncate in range", "window too small for the script"); return; }
                sequence = _a.Take(Math.Min(_a.Length, limit)).ToArray();
            }
            List<int> cold = sequence.Length == _a.Length ? _cold : Cold(sequence, _s.DecodeTokens);
            int cached = sequence.Length + _s.DonateAfter - 1;
            int want = cached - Math.Min(_s.RewindTokens, _caps.RewindCapTokens);
            int rewindTo = ResumabilityRules.AlignDown(want, _caps.TruncationGranularity);
            Assert.True(rewindTo > 0 && rewindTo < cached, "the script's rewind target is degenerate");
            Assert.True(_rules.TruncationAllows(cached, rewindTo) && _rules.RewindWithinCap(cached, rewindTo),
                $"the rules refuse the script's own rewind {cached} -> {rewindTo}");

            if (Holders)
            {
                (string key, int length, List<int> generated) = DonatedHolder("rewind", sequence);
                Assert.Equal(cached, length);
                Assert.True(_pcm.CanMaterialize(key, length, rewindTo), $"CanMaterialize refused an in-range rewind {length} -> {rewindTo}");
                MaterializeOp op = Copies ? MaterializeOp.Clone : MaterializeOp.Donate;
                string id = Request("rewind");
                Assert.True(_pcm.TryMaterialize(new MaterializeRequest(op, key, id, length, rewindTo)), $"{op} with a rewind target refused");
                if (op == MaterializeOp.Donate) _keys.Remove(key);
                Assert.False(_paged.BindSequenceCache(id));
                // The executor's first-bind truncation (the pending truncation of §5.6).
                Assert.True(_arch.TryTruncateKVCache(rewindTo), $"first-bind truncation {length} -> {rewindTo} refused");
                AssertRewoundContinuation(sequence, generated, rewindTo, cold);
                Release(id);
                Ran($"truncate in range ({length} -> {rewindTo}, {op})");
            }
            else if (_caps.PrimaryResident)
            {
                _paged.RestorePrimaryCache();
                _arch.ResetKVCache();
                List<int> generated = Greedy(_arch.Forward(sequence), _s.DonateAfter);
                Assert.Equal(cold.Take(_s.DonateAfter), generated);
                if (_caps.Truncation == TruncationKind.WithinRingSlack)
                {
                    int beyond = cached - _caps.TruncationParameter - 1;
                    if (beyond > 0)
                        Assert.False(_arch.TryTruncateKVCache(beyond), $"a rewind past the ring slack ({cached} -> {beyond}) must be refused");
                }
                Assert.True(_arch.TryTruncateKVCache(rewindTo), $"primary truncation {cached} -> {rewindTo} refused");
                AssertRewoundContinuation(sequence, generated, rewindTo, cold);
                _arch.ResetKVCache();
                Ran($"truncate in range on the primary ({cached} -> {rewindTo})");
            }
            else
            {
                NotApplicable("truncate in range", "no end states and no primary resident");
            }
        }

        private void ConvertPrimary()
        {
            if (!_caps.AdoptPrimaryOnDisplacement) { NotApplicable("primary conversion", "AdoptPrimaryOnDisplacement=false"); return; }
            _paged.RestorePrimaryCache();
            _arch.ResetKVCache();
            List<int> generated = Greedy(_arch.Forward(_a), _s.DonateAfter);
            Assert.Equal(_cold.Take(_s.DonateAfter), generated);
            int length = _a.Length + _s.DonateAfter - 1;
            Assert.Equal(length, _diag.PrimaryCacheLength);
            string key = Key();
            Assert.True(_pcm.TryConvertPrimary(key, length, out PayloadFootprint fp), "TryConvertPrimary refused");
            _keys.Add(key);
            Assert.Equal(length, fp.Tokens);
            Assert.Equal(0, _diag.PrimaryCacheLength);
            Assert.Contains(key, _diag.RetainedPayloadKeys);
            string id = Request("converted");
            Assert.True(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, key, id, length, length)));
            _keys.Remove(key);
            Assert.False(_paged.BindSequenceCache(id));
            AssertContinuation(generated, _cold);
            Release(id);
            // The fresh primary the conversion left behind serves a new request exactly.
            _paged.RestorePrimaryCache();
            Assert.Equal(_cold, Greedy(_arch.Forward(_a), _s.DecodeTokens));
            _arch.ResetKVCache();
            Ran("primary conversion");
        }

        private void BatchedRelease()
        {
            if (!Holders) { NotApplicable("batched release", "no end states"); return; }
            var batch = new List<string>();
            if (_caps.CanCaptureCopy)
            {
                string id = Request("batch");
                Assert.True(_paged.BindSequenceCache(id));
                _arch.Forward(_s.SharedPrefix);
                for (int i = 0; i < 3; i++)
                {
                    string key = Key();
                    Assert.True(_pcm.TryCaptureCopy(id, key, out _));
                    batch.Add(key);
                }
                Release(id);
            }
            else
            {
                for (int i = 0; i < 2; i++)
                    batch.Add(DonatedHolder("batch", _s.SharedPrefix).Key);
            }
            foreach (string key in batch) _keys.Add(key);
            long before = _diag.DecodeGraphResets;
            _pcm.ReleasePayloads(batch.Append("pc:conformance:never-minted").ToArray(), ReleaseReason.Evicted);
            long resets = _diag.DecodeGraphResets - before;
            Assert.True(resets <= 1, $"a batched release of {batch.Count} payloads issued {resets} decode-graph resets (DEC-24: at most one)");
            foreach (string key in batch)
            {
                Assert.DoesNotContain(key, _diag.RetainedPayloadKeys);
                _keys.Remove(key);
            }
            before = _diag.DecodeGraphResets;
            _pcm.ReleasePayloads(batch.ToArray(), ReleaseReason.Evicted);
            Assert.Equal(before, _diag.DecodeGraphResets);
            Ran($"batched release ({batch.Count} payloads, {resets} reset)");
        }

        private void ExportImport()
        {
            if (!_caps.Persistable) { NotApplicable("export/import", "Persistable=false"); return; }
            string id = Request("export");
            Assert.True(_paged.BindSequenceCache(id));
            _arch.Forward(_s.SharedPrefix);
            string exported = Key();
            Assert.True(_pcm.TryCaptureCopy(id, exported, out _));
            _keys.Add(exported);
            Release(id);
            var file = new MemoryStream();
            Assert.True(_pcm.TryExport(exported, file), "export of a capture refused");
            Assert.True(file.Length > 0);
            byte[] bytes = file.ToArray();
            ReleaseKey(exported);

            string wrong = Key();
            Assert.False(_pcm.TryImport(wrong, P + 1, new MemoryStream(bytes), out _), "an import must describe the requested token count");
            Assert.DoesNotContain(wrong, _diag.RetainedPayloadKeys);

            string imported = Key();
            Assert.True(_pcm.TryImport(imported, P, new MemoryStream(bytes), out PayloadFootprint fp), "import refused");
            _keys.Add(imported);
            Assert.Equal(P, fp.Tokens);
            string clone = Request("imported");
            Assert.True(_pcm.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, imported, clone, P, P)));
            Assert.False(_paged.BindSequenceCache(clone));
            Assert.Equal(_cold, Greedy(_arch.Forward(_s.Suffix), _s.DecodeTokens));
            Release(clone);
            ReleaseKey(imported);
            Ran($"export/import ({bytes.Length} bytes)");
        }

        private void ReleaseEverything()
        {
            if (_keys.Count > 0) _pcm.ReleasePayloads(_keys.ToArray(), ReleaseReason.Reset);
            _keys.Clear();
            foreach (string id in _requests) _paged.OnSequenceReleased(id);
            _paged.RestorePrimaryCache();
            Assert.Empty(_diag.RetainedPayloadKeys);
            Assert.Equal(0, _diag.PrivateHolderCount);
            Ran("release leaves no keys");
        }

        // ------------------------------------------------------------------ helpers

        /// <summary>A holder that prefilled <paramref name="sequence"/> and generated
        /// <see cref="ConformanceSubject.DonateAfter"/> tokens, donated under a fresh key.</summary>
        private (string Key, int Length, List<int> Generated) DonatedHolder(string what, int[] sequence)
        {
            string id = Request(what + "-src");
            Assert.True(_paged.BindSequenceCache(id));
            List<int> generated = Greedy(_arch.Forward(sequence), _s.DonateAfter);
            int length = sequence.Length + _s.DonateAfter - 1;
            int privateBefore = _diag.PrivateHolderCount;
            string key = Key();
            Assert.True(_pcm.TryCaptureDonate(id, key, length, out PayloadFootprint fp), $"TryCaptureDonate({length}) refused");
            _keys.Add(key);
            Assert.Equal(length, fp.Tokens);
            if (_s.PayloadBytesKnown) Assert.True(fp.Bytes.TotalBytes > 0, "a donated end state charges bytes");
            Assert.False(_paged.HasFusedSequenceCache(id), "a donated holder is no longer the request's");
            Assert.Equal(privateBefore - 1, _diag.PrivateHolderCount);
            Assert.Contains(key, _diag.RetainedPayloadKeys);
            Assert.True(_pcm.CanMaterialize(key, length, length));
            _paged.OnSequenceReleased(id);   // the release after a donation frees nothing
            Assert.Contains(key, _diag.RetainedPayloadKeys);
            return (key, length, generated);
        }

        /// <summary>A cold run: a fresh cache, one prefill, <paramref name="n"/> greedy tokens.</summary>
        private List<int> Cold(int[] tokens, int n)
        {
            if (Holders)
            {
                string id = Request("cold");
                Assert.True(_paged.BindSequenceCache(id));
                List<int> output = Greedy(_arch.Forward(tokens), n);
                Release(id);
                return output;
            }
            _paged.RestorePrimaryCache();
            _arch.ResetKVCache();
            List<int> result = Greedy(_arch.Forward(tokens), n);
            _arch.ResetKVCache();
            return result;
        }

        /// <summary><paramref name="n"/> greedy tokens; the last one is not forwarded.</summary>
        private List<int> Greedy(float[] logits, int n)
        {
            var output = new List<int>(n);
            for (int i = 0; i < n; i++)
            {
                if (i > 0) logits = _arch.Forward(new[] { output[^1] });
                output.Add(Argmax(logits));
            }
            return output;
        }

        /// <summary>The bound cache holds the sequence plus <paramref name="generated"/> minus its last token.</summary>
        private void AssertContinuation(List<int> generated, List<int> cold)
        {
            Assert.Equal(cold.Take(generated.Count), generated);
            var stream = new List<int>(generated);
            // Forwarding the last generated token predicts the next one the cold run produced.
            stream.AddRange(Greedy(_arch.Forward(new[] { generated[^1] }), cold.Count - generated.Count));
            Assert.Equal(cold, stream);
        }

        private void AssertRewoundContinuation(int[] sequence, List<int> generated, int rewindTo, List<int> cold)
        {
            int[] cachedTokens = sequence.Concat(generated.Take(generated.Count - 1)).ToArray();
            float[] logits = _arch.Forward(cachedTokens.Skip(rewindTo).ToArray());
            // Re-forwarding the rewound tokens reproduces the last generated token and every one after it.
            List<int> tail = Greedy(logits, cold.Count - generated.Count + 1);
            Assert.Equal(cold.Skip(generated.Count - 1), tail);
        }

        private void Release(string requestId)
        {
            _paged.OnSequenceReleased(requestId);
            _requests.Remove(requestId);
        }

        private void ReleaseKey(string key)
        {
            _pcm.ReleasePayloads(new[] { key }, ReleaseReason.Evicted);
            _keys.Remove(key);
            Assert.DoesNotContain(key, _diag.RetainedPayloadKeys);
        }

        private static int Argmax(float[] logits)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > logits[best]) best = i;
            return best;
        }
    }
}
