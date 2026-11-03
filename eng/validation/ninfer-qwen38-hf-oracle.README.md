# NInfer Qwen38 HF tokenizer oracle

This CPU-only tool compares a complete `NinferQwen38ImportProbe` export with the
original HF tokenizer. It does not install packages, download models, launch
CUDA, alter a factory, apply generation settings, or change original exports.
Invocation without `--execute` only prints a plan.

```sh
python3 eng/validation/ninfer-qwen38-hf-oracle.py \
  --probe-output /absolute/actual-import-probe-output \
  --resources /absolute/original-six-resources/frontend \
  --cases-file eng/validation/NinferQwen38ImportProbe/tokenizer-cases.json \
  --expect-probe-result-sha ACTUAL_RESULT_JSON_SHA256 \
  --out artifacts/ninfer-hf-oracle-FRESH.json \
  --execute
```

The existing Python environment must already contain `tokenizers`. Missing
dependencies produce **NOT_RUN**, not a passing oracle. The tool loads the
local, SHA-pinned `tokenizer.json` with `Tokenizer.from_file`, not a network or
`from_pretrained` call. It records the Python executable, package version and
available package/native-extension file hashes. Package identity does not
recertify an external environment or the C# build.

Both `add_special_tokens=False` and `True` are explicitly compared. Decode uses
`skip_special_tokens=False`, because the HF API's default skips special tokens.
The low-level Tokenizers decoder has no Transformers whitespace-cleanup layer;
no unsupported `clean_up_tokenization_spaces` keyword is passed. These API
semantics are documented in the [official Tokenizers reference](https://huggingface.co/docs/tokenizers/v0.20.3/en/api/tokenizer).

The input gate binds the operator-supplied `result.json` SHA, its complete public
reader/plan/frontend result, the original six resource SHA/size identities,
the fixed 22-case input SHA, and every export file SHA. Case identities, typed
IDs/flags, complete ordered token-ID matrices, per-token byte hashes, all-token
framing, and the 256 single-byte entries are checked. Duplicate, missing,
out-of-range or padded head IDs cannot produce a pass. The original input files
must remain byte-identical throughout analysis.

Qualification is explicitly divided:

- **Encode/decode:** 22 exported C# input sequences, with exact HF encode IDs for
  both special-token settings and exact HF decode of the actual candidate IDs.
  Equal decoded text cannot hide different BPE IDs.
- **Raw token bytes:** all valid IDs **0..248076**, compared against the original
  HF token strings through an independently implemented GPT2 byte inverse.
  This includes byte zero. Lossy single-token UTF-8 decoding is not used to
  reconstruct raw bytes.
- **Added vocabulary:** HF atomic matching and skip/retain behavior for all 33
  added tokens, distinguishing 21 `special=true` tokens from 12 ordinary added
  tokens. The C# export measures its special count; its complete runtime special
  ID list is not exported and must not be presented as directly measured here.

`complete=true` means all selected comparisons finished. `oracle_passed=true`
additionally requires every comparison to pass; mismatches are retained and
never netted against successful cases. Structural corruption is ERROR. Output
must be fresh, physically nonaliased, in an ignored `artifacts/` or
`docs/validation/` path, and outside the read-only input directories.

This is not an exhaustive text suite, full-model/BF16 oracle, inference support,
program quality, or performance result. C# `Decode` is directly checked for the
22 exported sequences; the full-vocabulary matrix measures `AppendTokenBytes`,
not every arbitrary sequence of decoded IDs. The 248320 head rows remain
separate from the 248077 vocabulary entries. No padded words or sampler masking
policy are invented. Original artifact/assembly identity remains an external
controller responsibility; this tool does not reread 19GB of weights.

Run the self-contained CPU contracts with:

```sh
python3 eng/validation/test_ninfer_qwen38_hf_oracle.py -v
```

Those contracts use a labelled tiny mock HF implementation. They test rejection
and scoring/binding logic, and do not count as executing the real HF oracle.
