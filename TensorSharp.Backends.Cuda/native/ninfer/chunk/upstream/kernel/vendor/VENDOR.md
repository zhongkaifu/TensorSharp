# The vendored reference subtree

Everything under `kernel/vendor/` except this file is a **verbatim copy** of a
file from the reference engine (ninfer) at one pinned commit. Nothing here is
written or edited by hand. That is the whole point of ADR 0010: a port claim
must be diffable, so "we ported the reference's kernel" means *this exact file
is the reference's file*, and a script proves it.

- Policy: `docs/adr/0010-vendored-reference-kernels.md`
- The exemptions: `docs/adr/0031-vendored-kernel-bottleneck-exemption.md` — a
  kernel *measured* as the bottleneck may carry a recorded patch, or be
  replaced by our own implementation in `kernel/src/` and leave the subtree —
  and `docs/adr/0037-a-vendored-file-may-carry-a-correctness-patch.md` — a file
  with a *demonstrated* correctness bug may carry a recorded patch, with a test
  that fails without it. Read both before editing anything here; nothing below
  changes.
- Attribution: `kernel/NOTICE` (the subtree is Apache-2.0; `LICENSE` is
  vendored alongside the code)
- Spec: `docs/specs/runtime/01-device-resident-forward.md` (GitHub #36)

## The manifest

`kernel/vendor/manifest.json` is the source of truth for what is vendored.

```json
{
  "reference": { "repo": "…", "branch": "…", "commit": "…", "default_path": "…" },
  "vendor_root": "kernel/vendor",
  "files": [
    { "path": "src/core/arena.h", "sha256": "…" },
    { "path": "src/core/arena.cu", "sha256": "…",
      "patch": { "diff": "patches/src/core/arena.cu.diff", "sha256": "…", "reason": "…" } }
  ]
}
```

- `reference.commit` is the **only** revision the hashes describe. Vendoring
  from any other checkout state is a hash mismatch, not a merge.
- `path` is shared by the reference and the leaf: `src/core/arena.h` in the
  reference is `kernel/vendor/src/core/arena.h` here. Vendored files keep the
  reference's namespaces and include paths (`#include "core/arena.h"`,
  `#include "ops/common/math.cuh"`), so a `diff` against the source is trivial
  and a reference update is a re-run of the script — never a re-port.
- `sha256` is the file's content hash **in the reference**. A vendored file
  must be byte-identical to it unless the entry records a `patch`.
- `.gitattributes` marks the subtree `-text` so git never rewrites the line
  endings the hashes describe.

## The script

`scripts/vendor-ninfer.ps1` wraps the `vendor-ninfer` binary of
`crates/vendor`. The logic lives in a Rust crate rather than in the script so
that `cargo test` covers it (`docs/agents/testing.md`).

| Command | What it does |
|---|---|
| `verify` | Every listed file exists under `kernel/vendor/` and hashes to what the manifest expects. With a reference checkout present, also checks the reference still carries the pinned content. **This is the byte-identical check.** |
| `sync` | Copies every listed file from the reference into the leaf. Verifies the reference against the pinned hashes *first* and copies nothing on a mismatch, so a wrong checkout cannot half-overwrite the subtree. Files with a recorded patch are kept unless `--force-patched`. |
| `repin` | Recomputes the hashes from the checkout on disk. Used when moving to a new reference commit — set `reference.commit` in the same change. |
| `record-patch PATH --reason TEXT` | Writes the local edit out as a diff under `patches/` and records its patched hash. |

Options: `--reference PATH` (defaults to `reference.default_path`, or
`$IGNIS_NINFER_REFERENCE`), `--manifest PATH`. Exit codes: `0` clean, `1` a
verification finding, `2` a usage or I/O error. `verify` works without a
reference checkout, so it runs from a bare clone.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/vendor-ninfer.ps1 verify
```

## The patch policy

A vendored file may not be edited. When the leaf genuinely needs a local
change, it is **recorded**, never silently applied:

1. edit the file under `kernel/vendor/`;
2. `scripts/vendor-ninfer.ps1 record-patch <path> --reason "<why>"` — this
   writes `patches/<path>.diff` and sets the entry's `patch.sha256` to the
   patched content's hash;
3. commit the diff with the file.

From then on `verify` expects the *patched* hash locally and the *reference's*
hash upstream, so both an unrecorded edit and upstream drift are caught. `sync`
will not overwrite a patched file unless it is told to.

A patch is admitted for one of two reasons, and its `reason` says which: a
measured bottleneck (ADR 0031), or a demonstrated correctness bug (ADR 0037),
in which case the `reason` also names the test that fails without the patch.
One patch has neither reason and says so: the W8 dispatch's arms for the
image model's vision tower (`src/ops/linear/w8/w8_dispatch.cpp` and its test
cases, spec image-model/03, GitHub #320), a registry extension sanctioned as
that ticket's one vendor edit. Its `reason` names the new problems and the
test cases that throw without it; whether such extensions become a third
standing reason is the owner's call. Its K-1152 arms (the tower's qkv,
output and fc1) reach the MMA route through an ADR 0031 patch of
`src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh` (GitHub #324): a row of scales
that is a multiple of 8 bytes but not of 16 is staged in two 8-byte halves.
The tower's profile opened it (those linears on SIMT were 73% of the
ten-image encode's kernel time) and shows the move (2.68 s to 0.43 s for
the ten-image tower).

### Ignis-patched behaviours

A correctness patch makes the behaviour it changes **Ignis patched**: ignis
deliberately computes something the reference does not. Every other vendored
behaviour is **reference parity**. Each Ignis-patched behaviour is one row
here, so a comparison with the reference can tell which side of a difference
it is looking at (ADR 0037).

| Behaviour | The reference | ignis | Test that tells them apart | Issue |
|---|---|---|---|---|
| The hq-e8-2b prompt route's residual ring, with the window on (`src/ops/launcher/gqa_attention_prefill.cu`, spec runtime/07) | Appends the chunk, then attends: the append rewrites the ring slots of the 512 keys before the chunk, so those keys are served the rows of chunk keys up to ~1,000 positions later | Attends, then appends: every key before the chunk is served its own ring row; after the call the ring and the codes are what the reference leaves | `ignis_kernel_hq_route_agreement_test`, the prefill-after-history arms (exact-window agreement with BF16, bit-exact causality; dense, masked, banded) | #258 |
| The A1 capacity query for a banded hq-e8-2b prompt call, past one 262,144-key scratch band (`src/ops/wrapper/gqa_attention.cpp`) | Counts the carry state as raw `(2 * head_dim + 8) * q_heads * width` bytes, while the op bumps its three tensors at 256-byte alignment: at a width with one key split and `width % 8 != 0` (321..448 and 769..896 on a 170-SM card) the answer is short by the padding after `carry_m`, and an arena sized by it throws `bad_alloc` mid-layer | Sizes the carry through `WorkspaceLayoutBuilder`, as it already sizes the split partials; the op itself computes nothing differently | `ignis_kernel_gqa_workspace_capacity_test`, the banded arms (every prompt width 17..1024 against a replay of the op's allocations) | #296 |

Anything that is not a verbatim vendored file — including a file we edit
beyond a recorded patch — is our own implementation and carries no provenance
claim (ADR 0010). The leaf's own code lives in `kernel/src/`,
`kernel/include/` and `kernel/tests/`, never here.

## What is vendored today

P1-06 (GitHub #42) vendors the substrate the op families are built on:

- **core** — `dtype`, `tensor`, `arena`, `device`, `layout`, the weight
  descriptor, the PDL helper and the NVTX headers. Built as the static library
  `ignis_vendor` by `kernel/CMakeLists.txt`, with `kernel/vendor/src` as the
  include root.
- **ops common** — all of the reference's `ops/common`: `math`, `memory`,
  `mma`, `rowsplit_mma`, `rowsplit_grouped_mma`, `warp`, `bf16_vector`,
  `sampling_workspace`, `token_slices`. `rowsplit_grouped_mma.cuh` includes
  the q4/q5 rowsplit storage headers, so those two are vendored with it — the
  substrate has no dangling include even though this model uses neither
  quantization.
- **the op-test harness** — `tests/ops/op_check.h` and `tests/ops/op_tester.h`,
  the reference's error-record convention, used by
  `kernel/tests/ignis_kernel_op_tests` (CTest). `kernel/vendor/tests` is that
  executable's second include root, because the harness includes itself as
  `"ops/op_tester.h"`.

P1-07 (GitHub #43) adds the norm and glue op family — the leaf's first vendored
op family with a public API:

- **norms + glue** — `rmsnorm` (plain and gated, one kernel/launcher/wrapper
  file each), `l2norm`, `residual_add`, `silu_mul`, `sigmoid_mul`: each op's
  kernel (`src/ops/kernel`), launcher (`src/ops/launcher`), wrapper
  (`src/ops/wrapper`) and public header (`include/ninfer/ops`, the leaf's
  first vendored include root — `kernel/vendor/include`). The wrapper and
  test `.cpp` files are plain host C++ that declare a `cudaStream_t` and (in
  the tests) call the CUDA runtime API directly, so `kernel/CMakeLists.txt`
  now depends on `CUDAToolkit` for its include/link paths — the `.cu` files
  never needed this because nvcc supplies them itself.
- **their reference tests** — `tests/ops/test_rmsnorm.cpp`,
  `test_gated_rmsnorm.cpp`, `test_l2norm.cpp`, `test_residual_add.cpp`,
  `test_silu_mul.cpp`, `test_sigmoid_mul.cpp` and the shared
  `tests/ops/norm_test_common.h`, each built as its own CTest executable
  (`ignis_<op>_test`) since a reference test file brings its own `main()` and
  cannot share a binary with another. Cases already exercise real 27B widths
  (hidden 5120, GDN norm 128/head, MLP intermediate 17408).

P1-08 (GitHub #44) adds the second op family: **embedding** (dense BF16,
Q6G64_F16S, W8G32_F16S, FP8_E4M3FN_ROW_BF16S gather —
`ops/kernel/embed_gather.cuh`, `ops/launcher/embed_gather.{h,cu}`,
`ops/wrapper/embedding.cpp`, its `ninfer/ops/embedding.h` public header, and
the FP8 geometry validator `ops/linear/fp8/fp8_format.{h,cpp}` the wrapper
dispatches through) and **argmax** (`ops/kernel/argmax.cuh`,
`ops/launcher/argmax.{h,cu}`, `ops/wrapper/argmax.cpp`, `ninfer/ops/argmax.h`),
each with its reference op test (`tests/ops/test_embedding.cpp`,
`tests/ops/test_argmax.cpp`) built by the same `ignis_<op>_test` CTest loop as
P1-07 (`ignis_embedding_test`, `ignis_argmax_test`) — without the reference's
`SKIP_RETURN_CODE 77` (ADR 0006: a missing GPU fails here, never skips).

P1-09 (GitHub #45) vendors the NVFP4 linear op family: `nvfp4_codec.cuh`
(E2M1/E4M3 decode), `nvfp4_config.h` (geometries, schedules, the registered
`[N,K]` problems), `nvfp4_format.{h,cpp}` (the `blockscale-k16-m128x4-v1`
weight-layout validator), `nvfp4_dispatch.{h,cpp}` (A16 vs. W4A4 route
selection), `nvfp4_gemv.{cuh,cu}` / `nvfp4_small_t.{cuh,cu}` /
`nvfp4_launch.h` / `nvfp4_output.cuh` (the decode and small-T A16 kernels and
their launchers), and the large-T W4A4 route in full —
`nvfp4_w4a4_plan.h`, `nvfp4_w4a4_mma.cuh` / `nvfp4_w4a4.cu` (the tensor-core
MMA route) and `nvfp4_w4a4_tma.{cuh,cu}` / `nvfp4_w4a4_tma_launch.h` (the
warp-specialized SM120 TMA route, which needs the CUDA driver API directly for
`cuTensorMapEncodeTiled`; since GitHub #329 that call, and its
`cuGetErrorName`, resolve to our `kernel/src/cuda_driver_shim.cu`, which
forwards them to the driver through the CUDA runtime, so nothing links the
driver's library and the vendored file is unchanged). Two
small epilogue/output headers from *other* op families are vendored alongside
it because `nvfp4_w4a4_tma.cu` is the shared W4A4 GEMM kernel those families
reuse: `ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh` and
`ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh` — both self-contained
structs over the already-vendored `ops/common/memory.cuh`, with no other
dependency. The public header `include/ninfer/ops/linear.h` is vendored too,
under a new `kernel/vendor/include` root: unlike the reference's
`src/ops/linear/linear.cpp` (not vendored — it dispatches every weight qtype
at once, pulling in op families this ticket does not vendor), the header
itself has no such dependency, so it is byte-identical here and the leaf's own
thin dispatcher (`kernel/src/linear.cu`, not vendored) implements it, routing
`QType::NVFP4` to the vendored `detail::nvfp4_dispatch` and throwing a clear
error naming P1-10 (#46) for every other qtype. The reference's own NVFP4 a16
linear test — `tests/ops/linear/test_nvfp4_a16.cpp` plus its shared harness
`linear_test_common.{h,cpp}` and the packed-weight fixture generator
`tests/ops/quantized_weight.h` — is vendored and built as the CTest target
`ignis_kernel_nvfp4_linear_tests` (its own executable: the reference's test
file brings its own `main()`, so it cannot be another source in
`ignis_kernel_op_tests`). The large-T W4A4/TMA sources compile as part of
`ignis_vendor` but are not exercised by this test; their own reference test
(`test_nvfp4_a4.cpp`) is out of scope until G2.

P1-10 (GitHub #46) vendors the BF16 and W8G32 linear families:

- **BF16 linear** — `src/ops/linear/bf16/` in full (GEMV, small-T, MMA) plus
  the public `include/ninfer/ops/linear.h` (`LinearPolicy`,
  `linear_workspace_capacity_bytes`, the two `linear()` overloads) that every
  linear family's dispatch header depends on.
- **W8G32 linear** — `src/ops/linear/w8/` in full (GEMV, small-T, MMA and
  split-K variants, the rowsplit storage/output decoders): `w8_dispatch`'s
  launch table spans geometries beyond this model's own (it is the
  reference's one dispatcher for the whole W8 family), so the family is
  vendored whole rather than picking out only the shapes G1 needs.
  `w8_dispatch.cpp` is **patched** (and `tests/ops/linear/test_w8_a16.cpp`,
  below): the six W8 problems of the image model's vision tower, see the
  patch policy above. `w8_rowsplit_gemm_mma.cuh` is **patched** for the
  tower's K-1152 problems (GitHub #324, ADR 0031), same place.
- **their reference op tests** — `tests/ops/linear/test_bf16_a16.cpp` +
  `tests/ops/direct_bf16_weight.h`, and `tests/ops/linear/test_w8_a16.cpp` +
  `tests/ops/linear/linear_test_common.{h,cpp}` + `tests/ops/quantized_weight.h`,
  each its own CTest executable (`ignis_kernel_op_test_bf16_linear`,
  `ignis_kernel_op_test_w8_linear`) — a vendored test file owns a `main()`, so
  it cannot be appended as another source of `ignis_kernel_op_tests`. Neither
  sets `SKIP_RETURN_CODE 77` (the reference's own CMake does, for CPU-only
  machines): under ADR 0006 a missing/busy GPU must fail the run, not skip it,
  so the byte-identical vendored `main()`'s `return 77` surfaces to CTest as a
  plain nonzero exit.
- The top-level `ops::linear` / `ops::linear_workspace_capacity_bytes`
  dispatch across qtypes is **ours**, not vendored (ADR 0010: it is glue that
  composes per-family dispatchers, not an op) — `kernel/src/ops/linear.cu`.
  It currently switches on `BF16_CTRL` and `W8G32_F16S`; `NVFP4` (P1-09/#45)
  extends the same switch.

P1-11 (GitHub #47) vendors the fused attention input projection and the fused
linear+residual output projection, NVFP4 and BF16 arms only:

- **attn_input_proj** — `src/ops/attn_input_proj/nvfp4/` (decode, small-T, the
  large-T W4A4 route) and `src/ops/attn_input_proj/bf16/` (decode, small-T,
  MMA) plus the public `include/ninfer/ops/attn_input_proj.h`. This is the
  fused q/k/output-gate/v projection: a single `[14336,5120]` parent weight
  (BF16_CTRL contiguous or NVFP4 block-scaled) split in the reference's row
  order — query `[0,6144)`, key `[6144,7168)`, output gate `[7168,13312)`,
  value `[13312,14336)` — into four independent output tensors, matching the
  24-query/4-kv-head × 256 head-dim split.
- **linear_add** — `src/ops/linear_add/nvfp4/` (decode, small-T, W4A4;
  `nvfp4_linear_add_epilogue.cuh` was already vendored alongside P1-09's
  shared W4A4 GEMM kernel) and `src/ops/linear_add/bf16/` (decode, small-T,
  MMA/aggregate-MMA) plus the public `include/ninfer/ops/linear_add.h`. This
  is the attention output projection fused with the residual add:
  `residual[:,t] += Linear(x,w)[:,t]` at `[5120,6144]` (the model's own
  geometry) and `[5120,17408]` (NVFP4 only, a second registered problem the
  reference's own test exercises).
- **their reference op tests** — `tests/ops/test_attn_input_proj.cpp`
  (**patched**: the reference's file is one combined test exercising Q4/Q5,
  BF16, NVFP4, FP8 and W8 arms in a single `main()`; this ticket vendors only
  NVFP4 and BF16, so the Q4/Q5, FP8 and W8 cases and their helpers are
  removed — `kernel/vendor/patches/tests/ops/test_attn_input_proj.cpp.diff`,
  recorded via `record-patch`) plus `tests/ops/input_projection_test_common.h`
  (unpatched, generic across arms); `tests/ops/linear_add/test_nvfp4.cpp` and
  `tests/ops/linear_add/test_bf16_a16.cpp` + `linear_add_test_common.{h,cpp}`
  (unmodified: the reference already splits `linear_add` into one test file
  per arm, so no patch is needed there). Each is its own CTest executable
  (`ignis_kernel_attn_input_proj_tests`, `ignis_kernel_nvfp4_linear_add_tests`,
  `ignis_kernel_bf16_linear_add_tests`).
- The top-level `ops::attn_input_proj` /
  `ops::attn_input_proj_workspace_capacity_bytes` and `ops::linear_add` /
  `ops::linear_add_workspace_capacity_bytes` dispatch is **ours**, not
  vendored (same reasoning as `kernel/src/linear.cu`: the reference's wrapper
  dispatches every registered qtype, including families this ticket does not
  vendor) — `kernel/src/attn_input_proj.cu` and `kernel/src/linear_add.cu`.
  Each switches on `BF16_CTRL` and `NVFP4` only and throws naming this ticket
  for every other qtype; the Q4/Q5 dual-weight `attn_input_proj` overload and
  the W8 single/companion overloads declared in the vendored header are never
  defined, matching that nothing in the trimmed test calls them.

P1-14 (GitHub #50) adds the GDN family:

- **causal_conv1d_silu** — `ops/kernel/causal_conv1d.cuh`,
  `ops/launcher/causal_conv1d.{h,cu}`, `ops/wrapper/causal_conv1d_silu.cpp`,
  `ninfer/ops/causal_conv1d_silu.h`: the depthwise causal width-4 conv fused
  with SiLU over the conv'd channels, plus its rolling-tap and B-way snapshot
  forms.
- **gdn_gating** — `ops/kernel/gdn_gating.cuh`, `ops/launcher/gdn_gating.{h,cu}`,
  `ops/wrapper/gdn_gating.cpp`, `ninfer/ops/gdn_gating.h`: the per-head decay
  gate `g = -exp(A_log)*softplus(a+dt_bias)` and update gate
  `beta = sigmoid(b)`.
- **gated_delta_net** — the whole
  `ops/linear_attention/gated_delta_net/` subtree: `common.{h,cuh}` (head
  mapping, chunk size, state dim), `launch.h` / `recurrent.{cuh,cu}` (the
  per-head fp32 128x128 recurrence, its distinct-state and B-way snapshot
  forms, and the replay-record entry point), `gated_delta_net.cpp` (the public
  wrapper: dispatches to the recurrent kernel for T=1 or a full chunk/tail
  split above it, L2-normalizing q/k once per call rather than per chunk when
  `normalize_qk` is set), `replay.cpp` (validation for the replay-record op),
  and `chunked/` in full (`common.cuh`, `prepare_wy_wu.{cuh,cu}`,
  `output.{cuh,cu}`, `state_passing.{cuh,cu}`, `launch.{h,cu}`) — the
  chunk-parallel WY/UT prefill route, compiled and exercised by the same
  reference test as the recurrent route (token counts that straddle the
  64-token chunk boundary). `ninfer/ops/gated_delta_net.h` is the public
  header for all three (recurrent, snapshot, replay-record) forms.
- **incidental**: `recurrent.cu` and `replay.cpp` also carry the reference's
  `gdn_replay_fold` implementation (the ReplaySSM state-pool fold), and
  `launch.h` declares it — one translation unit each, not split by op. Vendoring
  them whole therefore also pulls in `core/gdn_replay_records.{h,cpp}` and
  `ninfer/ops/gdn_replay.h`. `gdn_replay_fold` compiles as part of `ignis_vendor`
  but is not exercised by any test here; its own op family belongs to a later
  sequence-handle ticket.
- **their reference tests** — `tests/ops/test_causal_conv1d_silu.cpp`,
  `test_gdn_gating.cpp`, `test_gated_delta_net.cpp` (recurrent + chunked, real
  27B/35B-A3B geometries: 48 value heads / 16-32 qk heads x 128) and
  `test_gated_delta_net_replay_record.cpp`, plus the shared FP64 reference
  `tests/ops/gdn_ref.h` — each its own CTest executable via the same
  `ignis_vendored_op_tests` loop as the norm/glue family (P1-07): none of
  these ops dispatch across qtypes, so no leaf-side `kernel/src/*.cu`
  dispatcher is needed, unlike `linear` / `attn_input_proj` / `linear_add`.

P1-16 (GitHub #52) adds the sequence-state pools:

- **paged KV pool** — `core/paged_kv_cache.{h,cpp}`: pages, entitlements,
  block-table rows, selective zeroing, multi-pool reserve/resize bundles.
- **linear-attention state pool** — `core/linear_attention_state.{h,cpp}`:
  per-slot recurrent fp32 state + BF16/FP32 conv taps, copy/zero/pack/unpack.
- **ring bits** — `core/kv_ring_bits.{h,cu}`: the hq-e8-2b residual-ring
  validity-word helper, vendored with its public header even though no
  current op enables the feature (kept dangling-include-free like q4/q5
  above).
- their reference tests, `tests/test_kv_cache.cpp` and
  `tests/test_state_store.cpp`, each its own CTest executable
  (`ignis_vendor_kv_cache_test`, `ignis_vendor_state_store_test`) since each
  is its own `main()`.
- the leaf's own `ignis_paged_kv_page_budget` (`kernel/include/ignis_paged_kv_budget.h`,
  `kernel/src/paged_kv_budget.cu`) — not vendored, ours: reports page count /
  bytes for a VRAM budget and plane geometry, routed through
  `plan_paged_kv_pool` so it can't drift from what the pool itself allocates.

P1-15 (GitHub #51) vendors the GQA attention family: positions, RoPE, the
fused q/k norm+RoPE, paged KV addressing, KV append/prefix-append, and the
GQA attention kernels (bf16 decode + prefill launchers and routes gated here;
i8/hq launchers vendored so the family compiles, untested until a later
ticket):

- **position** — `ops/kernel/position.cuh`, `ops/launcher/position.{h,cu}`,
  `ops/wrapper/position.cpp`, `include/ninfer/ops/position.h`
  (`fill_i32_positions`, `offset_i32_positions`).
- **rope** — `ops/kernel/rope.cuh`, `ops/launcher/rope.{h,cu}`,
  `ops/wrapper/rope.cpp`, `include/ninfer/ops/rope.h` (the linear/vision
  frequency tables and the split-half NeoX rotation, Text 1-D / MRoPE /
  Vision modes).
- **qk_norm_rope** — `ops/launcher/qk_norm_rope.{h,cu}`,
  `ops/wrapper/qk_norm_rope.cpp`, `include/ninfer/ops/qk_norm_rope.h`: the
  fused per-head q/k RMSNorm + RoPE (no separate kernel header — the kernel is
  inline in the launcher `.cu`).
- **paged_kv_address** — `ops/kernel/paged_kv_address.cuh`: the paged
  block-table addressing helpers shared by every GQA attention route (no
  public API of its own, no dedicated test — exercised through
  `gqa_attention`).
- **gqa_attention** — `ops/kernel/gqa_attention_geometry.cuh`,
  `gqa_attention_kv_quant.cuh`, `gqa_attention_decode.cuh` +
  `_decode_{bf16,i8,hq}.cuh`, `gqa_attention_prefill_common.cuh` +
  `_prefill_{bf16,i8,hq}.cuh`, `hq_codec.cuh`; the launcher dispatch
  `ops/launcher/gqa_attention.h` (detail declarations) and
  `gqa_attention_{decode,prefill}.cu` (route selection) plus the per-dtype
  partial-kernel translation units `gqa_attention_decode_{bf16,i8}.cu`,
  `gqa_attention_decode_hq_{27,35}.cu` + `_decode_hq_routes.cuh`,
  `gqa_attention_prefill_{bf16,i8}.cu`, `gqa_attention_prefill_hq_{27,35}.cu`
  + `_prefill_hq_routes.cuh`; the wrapper `ops/wrapper/gqa_attention.cpp` and
  public header `include/ninfer/ops/gqa_attention.h`.
  (`gqa_attention_prefill.cu` is **patched** since GitHub #258 — the first
  Ignis-patched behaviour, in the table above.) Unlike
  `attn_input_proj`/`linear_add` (P1-11), the vendored wrapper *is* the
  complete public API — every cache dtype (BF16, I8, hq-e8-2b/U8) is
  vendored and compiles, so no leaf-side dispatcher (`kernel/src/*.cu`) is
  needed for this op.
- **kv_cache_append_prefix** — `ops/kernel/kv_cache_append_prefix.cuh`,
  `ops/launcher/kv_cache_append_prefix.{h,cu}`,
  `ops/wrapper/kv_cache_append_prefix.cpp`,
  `include/ninfer/ops/kv_cache_append_prefix.h`: device-selected exact K/V
  prefix commit, both the paged overload (this model's route) and the
  DFlash-lane cyclic overload (out of scope — MTP/DFlash2/ReplaySSM is G5 —
  vendored only because the header declares both in one file). Needs
  `core/cyclic_kv_cache.{h,cpp}` (new: the cyclic cache view/layout type),
  vendored purely to keep the header self-contained, same reasoning as
  P1-16's dangling-include-free `kv_ring_bits`/q4/q5 headers.
- **their reference tests** — `tests/ops/test_position.cpp`,
  `test_rope.cpp`, `test_qk_norm_rope.cpp` (unpatched, BF16-only in the
  reference itself), each built by the `ignis_<op>_test` CTest loop
  alongside the P1-07/P1-08 entries; `tests/ops/test_gqa_attention.cpp`
  (**patched**: the reference file exercises both BF16 and I8 cache dtypes
  across two head geometries; this ticket gates only the bf16 routes, so the
  I8 dtype arms and their key-split/range test cases are removed —
  `kernel/vendor/patches/tests/ops/test_gqa_attention.cpp.diff`, recorded via
  `record-patch` — leaving both registered head geometries (24q/4kv and
  16q/2kv, both ×256) at BF16, including multi-page paged-KV mappings)
  built as `ignis_kernel_gqa_attention_tests`; `tests/ops/test_kv_cache_append_prefix.cpp`
  (unmodified) built as `ignis_kernel_kv_cache_append_prefix_tests`.

P1-13 (GitHub #49) vendors the fused MLP gate_up projection with its SiLU-mul
epilogue, NVFP4 arm only:

- **linear_swiglu** — `src/ops/linear_swiglu/nvfp4/` (decode, small-T, the
  large-T W4A4 route and its own warp-specialized TMA kernel — unlike
  `linear_add`, this op's W4A4 GEMM epilogue is entirely private to it, so no
  shared epilogue header is vendored alongside it) plus the public
  `include/ninfer/ops/linear_swiglu.h`. Math: `gate_up = Linear(x,
  gate_up_weight)`; `out[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t]` where
  `M = gate_up_rows/2`; gate rows `[0,17408)` precede their matching up rows
  `[17408,34816)` at the model's `[34816,5120] -> [17408,T]` geometry
  (`Nvfp4MlpGateUpGeometry`, already registered by P1-09's
  `nvfp4_config.h`). The large-T route (`Nvfp4LinearSwiGluRoute::
  LinearW4A4Post`, inside the vendored plan) falls back to the already-vendored
  `ops::linear` (P1-09/P1-10) plus `ops::silu_mul` (P1-07) rather than its own
  fused kernel, so it compiles with no further vendoring; its own reference
  test is deferred to G2, same as P1-09's W4A4/TMA route.
- **its reference op test** — `tests/ops/linear_swiglu/test_nvfp4.cpp`
  (**patched**: the reference's file exercises both the A16 route this ticket
  vendors and the A4/large-T route deferred to G2 in one combined `main()`;
  the A4 case and its token list are trimmed —
  `kernel/vendor/patches/tests/ops/linear_swiglu/test_nvfp4.cpp.diff`, recorded
  via `record-patch`) plus `tests/ops/linear_swiglu/linear_swiglu_test_common.
  {h,cpp}` (unpatched, generic across arms). Its own CTest executable
  (`ignis_kernel_nvfp4_linear_swiglu_tests`).
- The top-level `ops::linear_swiglu` / `ops::linear_swiglu_workspace_capacity_
  bytes` dispatch is **ours**, not vendored (same reasoning as
  `kernel/src/linear.cu`: the reference's wrapper dispatches every registered
  qtype, including families this ticket does not vendor) —
  `kernel/src/linear_swiglu.cu`. It switches on `NVFP4` only and throws naming
  this ticket for every other qtype.

The remaining op family (the fused GDN projections: gdn_input_proj,
gdn_gating_proj) arrives with P1-12, adding its files to this manifest and
its reference op test to the leaf's test suite.

P2-03 (GitHub #85) qualifies the W4A4 activation-quantized route, turning on
the A4 cases that the earlier tickets trimmed:

- **linear A4 test** — `tests/ops/linear/test_nvfp4_a4.cpp` (new manifest
  entry, run unmodified): the reference's own `CallForm::Policy` `AllowA4`
  invocations at the model's attention-input `[14336,5120]`, GDN-input
  `[16384,5120]`, gate_up `[34816,5120]` and residual `[5120,6144]` /
  `[5120,17408]` geometries, built as `ignis_kernel_nvfp4_linear_a4_tests`
  (its own CTest executable, linking the leaf's `ops::linear` dispatcher
  exactly like the a16/w8 linear tests above).
- **linear_swiglu A4 case restored** — the patch recorded in P1-13
  (`patches/tests/ops/linear_swiglu/test_nvfp4.cpp.diff`, which trimmed the
  A4/large-T case until the W4A4 route was qualified) is dropped: the
  manifest entry reverts to the reference's unpatched hash, `sync` restores
  the unmodified test, and the A4 profile (`kA4Cases {5,48,49,128,1024}`)
  runs again in `ignis_kernel_nvfp4_linear_swiglu_tests`.
- The attention-input and GDN-input projection tests' A4 cases
  (`tests/ops/test_attn_input_proj.cpp` and `tests/ops/test_gdn_input_proj.cpp`)
  were already vendored with their A4 arms in P2-02 (GitHub #84) — those
  patches trim only the non-NVFP4 arms, so no manifest change is needed here.

P3-03 (GitHub #99) vendors the token sampler:

- **sampling** — `ops/kernel/sampling.cuh` (the single-block route),
  `ops/kernel/sampling_device.cuh` (shared candidate-selection/RNG
  primitives, also used by the reference's speculative acceptance, not
  vendored here), `ops/launcher/sampling.{h,cu}` (the multi-block-vs-single-
  block dispatch, reusing the already-vendored `ops/common/sampling_workspace.h`),
  `ops/wrapper/sampling.cpp` (the public `sample()` / `sample_and_scatter_hidden()`
  entry points — only `sample()` is called by this ticket, the hidden-state
  scatter overload is speculative-decoding-only, G5) and the public header
  `include/ninfer/ops/sampling.h` (`SamplingConfig`, `SamplePurpose`,
  `sampling_workspace_capacity_bytes`). RNG is counter-based
  (`sampling_uniform(seed, position, purpose, sub)`, a pure function of its
  inputs — no mutable device RNG state), so a sequence's determinism rests on
  the `(seed, logical_position)` pair the leaf passes each call, not on any
  stored generator state. Greedy (`temperature <= 0`) is bit-identical to
  `argmax()` by the vendored contract, so the leaf's program now calls
  `sample()` unconditionally in place of `argmax()` for every sampled
  position; the standalone `argmax()` op stays vendored and in use only by
  the degenerate G1 program (`ignis_prefill`/`ignis_decode`, test-only).
- **its reference op test** — `tests/ops/test_sampling.cpp` (unmodified: an
  independent CPU argmax oracle for the greedy branch and an FP64
  mathematical-distribution oracle for the stochastic branch, never the
  device RNG reproduced or another production path used as a golden), built
  as its own CTest executable (`ignis_kernel_sampling_tests`).
- The leaf's own program-layer wiring (`kernel/src/step.cu`: building
  `SamplingConfig` rows from the ABI's `ignis_sampling_params`, staging them
  and the logical-position array in stable device buffers for decode-graph
  replay, and the per-slot penalty-count buffer in the sequence pool) is
  ours, not vendored — `ops::sample`'s contract only supplies the op, not the
  ABI or the per-sequence state it reads from.

P4-03 (GitHub #119) vendors the hq-e8-2b row codec's op-level oracle:

- **its reference test** — `tools/test_kv/test_hq_codec.cu` plus
  `tools/test_kv/hq_dither_host.h` (host mirrors of the codec's subtractive-
  dither hashes it includes), both vendored verbatim, unpatched. It lives
  under `tools/test_kv/` in the reference rather than `tests/ops/`, which is
  why P1-15/#51's vendored `test_gqa_attention.cpp` carried no hq arm — the
  reference never put codec-level coverage there in the first place, only
  the attention-route coverage that ticket gated to bf16. The oracle itself
  is exhaustive FP64 2*E8 nearest-point checking of the device quantizer,
  device encode/decode round-tripped against an independent FP64 oracle,
  the per-row bit-budget invariant, cross-run determinism, 8-lane group-
  decode equivalence, an escalation/terminal-fallback census over a heavy-
  tailed corpus, the Rice-k invariant, and a synthetic k>0 general-fallback
  arm. `ops/kernel/hq_codec.cuh` (already vendored, P1-15/#51) is a header-
  only device header over the vendored substrate, so this needs no
  `ignis_kernel` link. Its own CTest executable, `ignis_hq_codec_test`,
  since it brings its own `main()` like every other tool/test file above.

P5-05 (GitHub #155) vendors the rest of the DFlash2 drafter's forward — the
ops `dflash2_propose_batch_impl` calls around the P5-01 drafter kernels:

- **swa** — `ops/launcher/swa.{h,cu}`, `ops/wrapper/swa.cpp`,
  `ninfer/ops/swa.h` and the kernel they launch,
  `ops/kernel/bidirectional_gqa_attention.cuh`: the symmetric non-causal
  sliding-window GQA over a lane's cyclic context plus its live query block
  (D=128, 32/8 heads, T=1..16, B=1..8).
- **prepare_masked_block** — `ops/kernel/prepare_masked_block.cuh`,
  `ops/launcher/prepare_masked_block.{h,cu}`,
  `ops/wrapper/prepare_masked_block.cpp`, `ninfer/ops/prepare_masked_block.h`:
  each lane's anchor-and-mask query block and its positions.
- **cast** — `ops/kernel/cast.cuh`, `ops/launcher/cast.{h,cu}`,
  `ops/wrapper/cast.cpp`, `ninfer/ops/cast.h`: the exact BF16 → FP32 widening
  the selector scores take.
- **their reference tests** — `tests/ops/test_swa.cpp`,
  `test_prepare_masked_block.cpp` and `test_cast.cpp`, unmodified, in the
  `ignis_<op>_test` CTest loop.
- Not vendored: `prepare_ragged_prefix` and `scatter`, which the reference uses
  to carry the verified columns' taps into the *next* round. ignis appends
  them to the window at the end of the same round instead
  (`kernel/src/dflash2_drafter.cu`, ours), so no tap outlives its round.

GitHub #177 (Vision 3/7) vendors the ops the vision tower runs on, at the same
pinned commit:

- **vision encoder ops** — `layer_norm`, `gelu` (tanh and exact modes),
  `add_bias`, `vision_pos_embed` (the bilinear 48×48 position-table add) and
  `vision_attention` (segmented, `cu_seqlens` per image): each op's kernel
  (`src/ops/kernel`), launcher (`src/ops/launcher`), wrapper
  (`src/ops/wrapper`) and public header (`include/ninfer/ops`).
- **scatter** — the same four files: it writes the encoder's `[5120, V]`
  output over a prefill span's placeholder columns.
- **prepare_ragged_prefix** — the same four files, vendored because the
  reference's `test_scatter.cpp` calls it; nothing in the leaf does.
- **Q4/Q5/Q6 row-split linear** — `src/ops/linear/q4/`, `q5/` and `q6/` in
  full (dispatch, GEMV, small-T / SIMT / MMA kernels, and the q6 storage
  header; the q4/q5 storage headers were already vendored). The leaf's own
  `ops::linear` dispatcher (`kernel/src/linear.cu`, ours, no provenance
  claim) gains the three arms, modelled on the reference's `linear.cpp`.
- Vision RoPE frequencies need nothing new: `rope_vision_frequencies` and the
  Vision rotation mode live in the already-vendored `rope` family (P1-15).
- **their reference tests** — `tests/ops/test_{layer_norm,gelu,add_bias,
  vision_pos_embed,vision_attention,scatter}.cpp`, unmodified, in the
  `ignis_<op>_test` CTest loop; `tests/ops/linear/test_{q4,q5,q6}_a16.cpp`,
  unmodified, each its own executable (`ignis_kernel_<q>_linear_tests`)
  linking the leaf's `ops::linear` like the W8 test.

## Updating to a newer reference commit

1. move the reference checkout to the new commit;
2. `scripts/vendor-ninfer.ps1 repin` and set `reference.commit`;
3. `scripts/vendor-ninfer.ps1 sync` (re-apply and re-record any patch);
4. `kernel/build.ps1 -Test` — the leaf's op tests are the acceptance.
