# Corrective Qwen video and counted long context — r4 Runtime derivative

All four scenario suites pass, but **the complete run remains failed** because the final whole-application directory check detects added runtime logs/uploads. Existing application files are unchanged, and model/head/media hashes and file metadata match before/after. Original failed records are retained in [qwen-http-r4](../qwen-http-r4/README.md); a fresh full integrity replay with external runtime storage remains required.

The production change preserves individual sampled frame timestamps in each video's prompt metadata while keeping the trained temporal-pair vision blocks and their mean timestamps. Previously, frames sampled at0 and2 seconds formed a block labeled1 second, losing the individual times. The model read codes17/86 correctly but answered1 when asked when the second frame appeared. The source correction has33 focused passing tests.

On the actual trained Qwen3.8 UD-Q2_K_XL checkpoint, physical GPUs0/1/5:

- API contract regression:32/32 pass.
- Ordered video read and timestamp follow-up: both turns pass in streaming and blocking modes; second-frame answer is2 seconds.
- Corrected long request: exact pinned body with **65,065 prompt tokens**,256 output budget and4 verification reserve. HTTP usage confirms65,065 prompt +48 completion tokens. All three retrieval values match, finish=stop,86.22seconds observed on a shared host.

The original67,154-token over-limit input is unchanged and still failed. This replacement had a separate launcher failure in the first campaign because the helper's model argument omitted the `.gguf` suffix present in the pinned body. The corrected helper argument preserves the exact wire body, token sequence and original expected values.

Runtime-only derivation and full application manifest are under [provenance](provenance/). Native remains r4 SHA256 `7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`. The Runtime replacement leaves the other production assemblies unchanged. Exact native and mapped managed files were checked between suites; see [profile](lifecycle/profile.json) and [owner](owner.json).

This covers one short ordered video and one long text retrieval request, not arbitrary video quality, video generation, trained speculative media support or quiet performance gates. The checkpoint has QSA indexer topK2048 with ratio4 on12 full-attention layers, but this retrieval test alone does not establish independent full-checkpoint QSA numerical parity. The helper's recorded `qsa_qualified=false` is retained.
