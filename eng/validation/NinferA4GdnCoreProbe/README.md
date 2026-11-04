# Fixed T8 W4A4 components

This standalone experiment changes only the TS-owned GEMM template schedule:
BN32/3 stages becomes BN8/2 stages. Only five actual T8 projection roles are
accepted, selected with `--shape`; default is `gdn-input`. Block256 and dynamic
shared memory76800/43520 bytes remain fixed. Production
files and existing probes are unchanged. The original Ignis implementation
informs the smaller activation tile; no foreign format or arithmetic is used.

```sh
dotnet run --project eng/validation/NinferA4GdnCoreProbe -- --self-test --shape gate-up
dotnet /absolute/NinferA4GdnCoreProbe.dll \
  --shape gate-up \
  --helper-module /absolute/old-ninfer-helper.fatbin --expect-helper-sha256 SHA \
  --legacy-module /absolute/actual-q35-kernels.fatbin --expect-legacy-sha256 SHA \
  --candidate-module /absolute/t8-candidate.fatbin --expect-candidate-sha256 SHA \
  --out /absolute/artifacts/a4-t8-component --timing
```

Actual driver parameter offsets/sizes and all loaded image SHA256 values are
verified. Three fixtures compare actual old32, same-source32 in the candidate
image, and new8 over two module lifetimes and A/B/A poisons. Every F32 GEMM
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
match actual old32 exactly; all rounded outputs and split partials then match
at every interval (split1 partials remain guards-only). Independent FP64 checks
cover only the explicitly recorded active-row witnesses before the consumer
in this dense timing fixture; uncomputed dots are never counted as FP64 checks.
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
