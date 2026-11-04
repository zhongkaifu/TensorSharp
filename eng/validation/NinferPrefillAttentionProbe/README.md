# Original BF16 prefill page-adapter component

This probe checks the fixed Q24/KV4/HD256 prefill adapter against the **unchanged
original Ignis kernel in the same image**, including every active BF16 output
and every active split F32 partial bit. Original64-token I32-page caches and
TensorSharp256-token pointer-page caches have independently permuted physical
pages and identical represented logical BF16 values. Unused pages/tokens contain
BF16 NaNs, and input allocations have guards and immutable hashes.

Eight fixed cases cover64/65/129/416/512 query rows, all original split counts,
63/64 and255/256 cache boundaries, a causal future-value sentinel, and the
measured late32K full/tail geometries. The bounded dense fixtures use periodicK
and nonperiodicV: an independent FP64 grouped-score/prefix-sum oracle evaluates
**every output** without quadratic CPU storage. A complete naive-key oracle
independently checks one output per case, including32K. Fixed prior attention
limits are abs0.015625 +rel0.005 against the ideal (BF16P is not narrowed in this
FP64 ideal); constant/sentinel controls have additional exact checks. This is a
scoped synthetic envelope, not a universal accuracy or model-quality claim.

The F32 query/gate bridge is checked separately: packedQ and coreBF16 must match
the directBF16 original, followed by finite gated output/reference checks. The
old TensorSharp kernel is an explicit pinned diagnostic control; its output
must satisfy the independent numeric gates, but need not equal original Ignis
bits. Original and candidate use BF16 K/V without lossy KV conversion.

```sh
dotnet build eng/validation/NinferPrefillAttentionProbe/NinferPrefillAttentionProbe.csproj -c Release
dotnet eng/validation/NinferPrefillAttentionProbe/bin/Release/net10.0/NinferPrefillAttentionProbe.dll --self-test
dotnet eng/validation/NinferPrefillAttentionProbe/bin/Release/net10.0/NinferPrefillAttentionProbe.dll --execute \
  --module /absolute/new.fatbin --expect-sha256 NEW_SHA \
  --legacy-module /absolute/old.fatbin --expect-legacy-sha256 OLD_SHA \
  --out /absolute/artifacts/fresh-prefill-components
```

Execution requires the pinned170SM campaignGPU and direct image paths. Two
module lifetimes and three output poisons per case are mandatory:16 records,
48 passes,144 original/candidate core submissions,462 total native submissions
including bridges/reducers/old controls. Every complete body output is retained;
differing split partials are retained on failure. Raw logical inputs, ideal
arrays, physical permutations, offsets and allocation hashes reconstruct input
layouts. Only one case is live at a time. The successful evidence budget is
bounded below1.9GiB; use fast ignored POSIX storage, then archive once.

`completed-result.json` is created once, with a SHA sidecar, only after CUDA
modules/allocations/context are disposed and every retained file is rehashed.
Progress `result.json` is not the authoritative completion artifact. The probe
uses the default stream only, prepares caches directly, and does not validate
append kernels, multiple sequences, Verify, sparse invalid rows, model quality,
or end-to-end performance. CPU self-test is not GPU validation.
