# Same-image BF16 attention schedule proof

This tool changes **only host launch geometry** for the existing
`ts_ninfer_attn_bf16_f32` symbol: control256threads/ceil(rows*6/128) versus
candidate128threads/ceil(rows*6/64). Both arms load the same pinned image and
use the same12argument ABI. Native arithmetic, weights, cache dtype, attention
geometry and gate epilogue remain the existing TensorSharp implementation.
There are no native source edits or new tactics.

The kernel assigns16query rows perwarp; both CTArow boundaries are multiples16.
Individual queries retain warp-local QK/PV MMA, exp2, accumulator and denominator
order. The smaller CTA can omit trailing keytiles that were entirely causally
masked for that query. Signedzero/zero products mean source reasoning alone is
insufficient: **every F32 output bit must equal the256thread control**.

The eight frozenR20cases, full independentFP64/analytic gates, shuffledTS256
physical cache layout, two module loads percase, three poisons, immutable input
hashes, guards and cross-lifetime/repeat byte checks are mandatory. Unused cache
tokens/pages contain BF16NaNs. All56regenerated fixture blobs must match the
SHA manifest in the pinned successfulR20component certificate before GPU use.
R20is reused only as fixture provenance, never as this schedule's GPU/quality
proof. Its failed end-to-end Ignis arithmetic candidate is not adopted here.

```sh
dotnet build eng/validation/NinferAttentionScheduleProbe/NinferAttentionScheduleProbe.csproj -c Release
dotnet eng/validation/NinferAttentionScheduleProbe/bin/Release/net10.0/NinferAttentionScheduleProbe.dll --self-test
dotnet eng/validation/NinferAttentionScheduleProbe/bin/Release/net10.0/NinferAttentionScheduleProbe.dll --execute \
  --module /absolute/unchanged-old.fatbin --expect-sha256 MODULE_SHA \
  --fixture-certificate /absolute/r20/completed-result.json --expect-certificate-sha256 R20_SHA \
  --out /absolute/artifacts/fresh-schedule-component
```

Default qualification is16records/48poison passes/**96actual kernel submissions**.
Both complete raw outputs are byte-compared and retained in SHA-addressed CAS;
every reused CAS object is byte-compared again, not merely hash-compared.
Repeated equal outputs share one object, avoiding duplicate input/output storage.
One fixture is live at a time. Successful full-output CAS is at most46.5MB.
Inputs reconstruct from unchanged fixture source plus56verified blob digests,
physical permutations, offsets, allocation hashes and the copiedR20certificate.

Only after this complete proof, an explicit separate timing invocation admits
the component certificate and rehashes all its complete CAS objects:

```sh
dotnet eng/validation/NinferAttentionScheduleProbe/bin/Release/net10.0/NinferAttentionScheduleProbe.dll --timing \
  --module /absolute/unchanged-old.fatbin --expect-sha256 MODULE_SHA \
  --component-certificate /absolute/schedule-component/completed-result.json --expect-certificate-sha256 COMPONENT_SHA \
  --out /absolute/artifacts/fresh-schedule-timing
```

Timing uses only the observed late32K geometries:512rows/past31744 and416rows/
past32256. Each has5ABBA rounds (256,128,128,256), six CUDA-event timed replays
perinterval, and one same-address warmup outside each interval. Full qualified
control bits, independent numeric checks, immutable inputs and guards are
checked after every interval. Total40intervals/240timed+40warmup submissions;
all raw outputs use CAS. Cache scope is same-address replay, not physicalcold.
A conservative predeclared kernel win requires >1.05median speedup and at least
4/5paired round wins for **both** cases. All timings are reported even if this
criterion fails; failure to beat it does not erase a numerical component pass.

Each successful mode creates `completed-result.json` once with a SHA sidecar
after all allocations/modules/context are disposed and raw evidence is rehashed.
Progress `result.json` is not authoritative. Defaultstream only; no append,
Verify, multiple sequences, model quality or end-to-end performance qualification.
CPU self-test is not GPU validation. FrozenR20tools/native sources are untouched.
