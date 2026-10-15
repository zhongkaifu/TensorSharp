# Pinned typed tensor diagnostics

`compare-typed-tensor-traces.py` validates completed tensor captures and compares
an explicit common token prefix. It requires Python 3.10+ and NumPy. It neither
loads a model nor executes a backend. The reader accepts the generic schema
`tensorsharp.tensor-trace.v1` and the initial producer schema
`tensorsharp.glm-trace.v1`; the comparison policy does not encode a model's answer,
routing, precision, admission or memory policy.

This is a localization tool. Retaining intermediate tensors can change graph
liveness, fusion and allocation. Matching tensors are **not** acceptance of model
quality, EOS behavior, performance, shared-ledger accounting or physical hard
limits. No numerical tolerance promotes a capture to a quality PASS. Existing
model-level gates and their thresholds must still run on the ordinary graph.

## Capture contract

Each new, empty absolute capture directory contains committed
`pass-000000.json` manifests and `pass-000000.tensor-0000.bin` data files.
Pass and tensor ordinals use six and four decimal digits. A producer publishes
each complete binary before publishing its manifest. Any `.incomplete` marker,
temporary file, orphan binary, non-regular file, path traversal or unpinned extra
file causes rejection. Freshness is established by the owned execution controller
before launch; it cannot be reconstructed from filenames or timestamps afterward.

A manifest records `schema`, `pass_id`, positive `model_id`, `kind`
(`trunk`, `batched`, `mtp`), `p0`, `nt`, `n_kv`, `slot_id`, actual
`sequences: [{slot_id, position}]`, `rows_limit`, `max_bytes`, `total_bytes`,
`endian` (`little` or `big`), `token_axis_overrides`, and ordered `traces`.
N1 captures must contain their actual one-element sequence identity and integer
`p0`; joint captures use null `p0` and explicit sequences. The reader validates
joint captures, but this comparator rejects comparing them until an explicit row
to sequence mapping is available.

`n_kv` is positive for `trunk` and `mtp`. A `batched` producer may emit zero when
one combined graph-wide KV extent is undefined; this is not a measurement of zero
KV allocation or usage. That sentinel requires null `p0` and exactly `nt` explicit
unique `(slot_id, position)` row identities. It does not enable joint-prefix
comparison, infer a KV extent, or establish physical-memory accounting.

Each trace records `ordinal`, `name`, `tag`, `layer`, `rank`, `type` and
`type_code` (F32/0 or I32/26), `ne[4]`, `nb[4]`, `source_bytes`,
`candidate_token_axes`, `token_axis`, `token_axis_source`, `shape[4]`,
`slice_start`, `token_count`, `logical_elements`, `bytes`, `file` and
`content_fnv1a64`. Ordinals are contiguous registration order. The
`(tag, layer, rank)` key is unique per pass.

`ne`, `nb` and `source_bytes` describe the **original** tensor, including original
strides. The reader checks every dimension's reachable byte envelope. The binary
contains **packed logical elements in ne0-first order**, with no original stride
padding. Its complete length is exactly `prod(shape) * 4`. Applying original
`nb` to the packed binary would reinterpret the data incorrectly. Endianness is
explicit, and every dumped F32 element must be finite.

The token axis must have original extent `nt`. `candidate_token_axes` enumerates
all such dimensions. A unique candidate is accepted with source `unique-extent`;
ambiguous candidates require source `override` and a matching explicit tag to
axis entry in `token_axis_overrides`. No candidate, guessed axis, partial file,
incorrect slice, or silent trace skip is accepted. The slice starts at zero and
clips precisely that axis to `min(nt, rows_limit)`.

Native `content_fnv1a64` uses the standard 64-bit FNV-1a offset
14695981039346656037 and prime 1099511628211 with unsigned wrap. It is
noncryptographic. The reader verifies it and separately verifies SHA256 against
the external run pin; it does not treat FNV as an identity certificate.

Optional `snapshot_checked_elements` and `snapshot_source` describe an independent
producer-side logical snapshot check. The count is either zero with null source,
or the full dumped `logical_elements`. A source descriptor contains `kind`,
`layer`, `rank`, `tensor_name` (possibly empty), `op`, `ne`, `nb`, and
`source_bytes`. Original and snapshot logical geometry must match. The tool
validates and preserves this metadata; native source-comparison execution is a
separate producer qualification, not proven by a JSON assertion alone.

## External run pins

After the owned process and its controller finish, freeze the capture and create
a separate JSON pin **outside** the capture directory. Record actual receipts;
do not fabricate successful exit status or change a probe's quality result.
Example field layout (hash strings below are placeholders):

```json
{
  "schema": "tensorsharp.tensor-trace-pins.v1",
  "capture_root": "E:\\validation\\teacher512\\trace",
  "fresh_directory": true,
  "controller_exit_code": 0,
  "producer_exit_code": 1,
  "expected_producer_exit_code": 1,
  "producer_binary_sha256": "<actual 64 lowercase hex>",
  "producer_build_receipt_sha256": "<actual 64 lowercase hex>",
  "probe_report_sha256": "<actual original probe report SHA256>",
  "execution_receipt_sha256": "<actual execution/cleanup receipt SHA256>",
  "model_identity_sha256": "<actual frozen model identity receipt SHA256>",
  "input_ids": [1, 2, 3],
  "files": {
    "pass-000000.json": "<actual SHA256>",
    "pass-000000.tensor-0000.bin": "<actual SHA256>"
  }
}
```

`capture_root` equals Python's resolved absolute directory string on the machine
running the comparator. The `files` map covers exactly every committed capture
file. The producer binary/build hashes bind the native implementation; the model
identity receipt binds model files and configuration; actual `input_ids` bind the
tokenized input history. The caller supplies trusted receipt hashes, not merely
model filenames. Keep original receipt files with the validation evidence.

The controller must exit zero after validating execution and cleanup. Producer
exit codes may be nonzero when explicitly expected by the controller, such as a
quality probe with an intentionally unknown task. The actual and expected codes
must match. The original report's `Passed=false` or unknown quality status stays
unchanged. A successful diagnostic controller never turns it into a quality PASS.

Use optional `--reference-pin-sha256` and `--candidate-pin-sha256` to bind run pins
to SHA256 values stored by an external trusted controller. Without those flags,
the tool records the run-pin SHA256 it read and reports that external pin SHA256
was not independently verified. It still checks all capture file hashes and
rehashes the complete inventory and both pins after comparison.

## Comparison and interpretation

```powershell
python eng/validation/compare-typed-tensor-traces.py `
  --reference E:/validation/teacher512/trace `
  --reference-pin E:/validation/teacher512/trace-pins.json `
  --candidate E:/validation/teacher774/trace `
  --candidate-pin E:/validation/teacher774/trace-pins.json `
  --prefix-tokens 512 `
  --output artifacts/example/typed-trace-comparison.json
```

The output must be new and outside both immutable capture directories. Store
generated evidence under ignored `artifacts/`, `docs/validation/`, or a separate
validation cache; never force-add captures or reports to Git.

Comparing N1 passes requires identical model identity, kind, starting `p0`,
non-token geometry and actual input IDs through `p0 + prefix_tokens`. The two
full prefill widths may differ. Future input IDs do not participate in the common
prefix comparison. The report records both pass extents and exactly the verified
history extent; pinning an entire 1343-token run does not claim that its first
512-token prefill pass has already consumed all 1343 tokens.

I32 comparisons require every selected logical ID to match exactly. F32 reports
bitwise and numerical difference counts, maximum/mean/quantile absolute errors,
signed mean, RMSE, relative L2 and cosine. Signed zero can differ bitwise with no
numerical error. The first divergent tag/layer is the first **observed registration
stage**, not proof of the underlying causal operation. Missing, extra or reordered
stages make the diagnostic incomplete and suppress the ordered-first claim.

`matched` and `localized` return exit zero to indicate a completed diagnostic.
`invalid` and `incomplete` return exit two. `numerical_quality_passed`,
`semantic_quality_passed` and `performance_accepted` are always null. Diagnostic
traces do not replace actual-model correctness or uninstrumented performance runs.

Reader bounds default to 256 MiB per tensor and 512 MiB per capture; override
`--max-tensor-bytes` and `--max-capture-bytes` explicitly for larger captures.
Metadata and run pins are each limited to 16 MiB. Multiple committed passes need
explicit `--reference-manifest` and `--candidate-manifest` selections. FNV/SHA and
error statistics require CPU reads; their wall time is not inference performance.

## Independent reader tests

```powershell
python -X utf8 -B -m unittest discover -s eng/validation/tests `
  -p test_compare_typed_tensor_traces.py -v
```

Tests build packed binary fixtures independently with `struct`, including
noncontiguous original strides, F32/I32, endian changes and a true 512-token common
prefix. They test malformed metadata, source bounds, path escapes, incomplete
captures, byte truncation, checksums/pins, token-axis ambiguity, context mismatch,
registration ordering and capture mutations. They execute no model or GPU.
