# Fixed T8 W4A4 components

This standalone experiment compares TS-owned GEMM schedules. Only five actual
T8 projection roles are accepted, selected with `--shape`; default is
`gdn-input`. The optional `--legacy-schedule bn32|bn8` selects the actual legacy
module's entry. Omitting it preserves the original BN32/3-stage baseline and
CLI. `bn8` instead compares the already accepted R34 BN8 entry with a new BN8
candidate, so its timing cannot reuse the earlier BN32→BN8 improvement.

| Legacy selection | Entry in pinned legacy module | Dynamic shared bytes | Recorded arm |
|---|---|---:|---|
| `bn32` (default) | `ts_q35_gemm_w4a4_32` |76800|`actual-old32`|
| `bn8` | `ts_q35_gemm_w4a4_8` |43520|`actual-old8`|

Block256, all projection geometries, quantization, consumers and mathematical
gates stay fixed. The candidate image's `same-source32` entry remains a
76800-byte numerical control; its `candidate8` entry remains43520 bytes.
Production and native source files are not changed by this tool option.

```sh
dotnet run --project eng/validation/NinferA4GdnCoreProbe -- --self-test --shape gate-up
dotnet run --project eng/validation/NinferA4GdnCoreProbe -- --self-test --shape gate-up --legacy-schedule bn8
dotnet /absolute/NinferA4GdnCoreProbe.dll \
  --shape gate-up \
  --legacy-schedule bn8 \
  --helper-module /absolute/old-ninfer-helper.fatbin --expect-helper-sha256 SHA \
  --legacy-module /absolute/actual-q35-kernels.fatbin --expect-legacy-sha256 SHA \
  --candidate-module /absolute/t8-candidate.fatbin --expect-candidate-sha256 SHA \
  --out /absolute/artifacts/a4-t8-component --timing
```

Actual driver parameter offsets/sizes and all loaded image SHA256 values are
verified. Three fixtures compare the selected actual old32/old8, same-source32
in the candidate image, and new8 over two module lifetimes and A/B/A poisons.
The selection, actual legacy symbol/shared size and honest arm label are recorded.
Every F32 GEMM
output and all used split partials must match bit for bit, then the entire
unchanged consumer output must match. Independent CPU quantization must match
every activation code/scale and row scale. The recorded active rows are adapted
to each role's dimension/tails, and GateUp includes
paired gate/up rows. Other rows are zero and explicitly checked. FP64 decoded
quantized dots and fixed gamma bounds cover the entire before-consumer synthetic
output; the binary fixture also
proves exact accumulation by a bounded binary lattice. Inputs, all guards,
counter reset, nondefault-stream ordering and cross-lifetime A/A are checked.

| Shape | N×K | Splits/grid | Unchanged consumer | Component kernels |
|---|---:|---:|---|---:|
| `gdn-input` |16384×5120|2 /128×1×2|BF16 round|162|
| `gate-up` |34816×5120|1 /272×1×1|F32 SwiGLU→BF16|162|
| `down` |5120×17408|8 /40×1×8|None; output stays F32|108|
| `output` |5120×6144|8 /40×1×8|None; output stays F32|108|
| `gqa-input` |14336×5120|2 /112×1×2|BF16 round|162|

Split1 partial and counter contents are unused and checked for guards only,
never called verified partials/reset counters or compared across poisons.
Their semantic result fields are null. After-consumer checks require
full old/same-source/new bits and finite values; BF16 format applies only to
Round/SwiGLU. No new independent SwiGLU numerical envelope is invented.

Optional timing is admitted only after all components pass. It uses a separate
dense pseudorandom weight payload at the selected role's full size. All full F32 outputs first
match the selected actual legacy exactly; all rounded outputs and split partials then match
at every interval (split1 partials remain guards-only). Independent FP64 checks
cover only the explicitly recorded active-row witnesses before the consumer
in this dense timing fixture; uncomputed dots are never counted as FP64 checks.
Only the selected legacy and `candidate8` are timing arms; `same-source32` is
never timed. For `--legacy-schedule bn8`, both the actual old and candidate
captured GEMMs must resolve their own module's `_8` function and43520-byte
shared-memory descriptor. This is an old8/new8 comparison, not old32/new8.
Both arms capture the role's complete quant→GEMM→consumer two/three-kernel
chain, sharing immutable input/weight addresses. Five ABBA blocks use64 replays
per interval and one symmetric outside warmup. Used split-K counters return to
zero without an unmeasured reset between replays; split1 counters remain
guards-only. Warm repeated weights may fit L2:
this is neither a physical-cold/DRAM measurement nor an end-to-end result.
The separate `predeclaredWin` requires>5% median improvement and4/5 paired wins;
numerically valid timing alone is not speed qualification.

All complete outputs use byte-checked SHA256 CAS. `completed-result.json` and
its sidecar are written once after graphs/modules/context dispose and CAS
rehashing. `result.json` is progress. GPU, model quality and end-to-end claims
require actual separate execution; CPU self-tests and skips do not establish them.
