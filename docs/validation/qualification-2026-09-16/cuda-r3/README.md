# Fresh CUDA build r3 — functional evidence, release blocked

This immutable intermediate snapshot compiled from scratch on the seven-A40 VM,
with physical GPUs 0, 1, 5 and 6 exposed to its tests. Other inference workloads
were active. It establishes functional results, not quiet performance acceptance.
Subsequent source changes require new validation.

The unchanged ggml checkout is `456172ec733a135778adcd32d00e576a58232e45`.
[Source identity](source.json) has SHA256
`649305ac55cf9a325ff82159db287f3c01d95498f4fab82cc22a498912123205`.
The fresh native library is
`b6cf00377fca4154f8d318677e201044bf93adfa97ef26b0d6742ce1d4cbe751`;
the compiled DeepSeek object is
`e8c2600854085092e7a60b332a7a807476fea87cb322d664619c57c1439cd483`.
[Build evidence](build.json) records all 294 object hashes, exact commands,
managed assembly hashes and successful before/after source/dependency audits.
The verbose compiler log remains on the VM at
`/workspace/ts-codex-20260916-r3/results/build1/native-build.log`.
The earlier stale-object run remains failed.

| Check | Actual result |
|---|---|
| Fresh native CUDA Release build | Passed |
| Server, test and benchmark Release builds | Passed |
| [DeepSeek C ABI boundary](c-abi-boundary.json) | Passed |
| [Native CTest](native-tests.xml) | 26 passed; seven- and eight-device cases unavailable in this four-device test allocation |
| Qwen MTP operator fixtures | All eight CPU/CUDA capacity/alignment combinations passed |
| Qwen target snapshot fixtures | CPU and CUDA passed |
| Qwen sparse-attention independent fixtures | CPU and CUDA passed |
| [Integrated portable managed suite](managed-portable.log) | 4,265 passed, four failed, three skipped |

The four managed failures are the multimodal request-isolation fixtures. The
new video cache constructor changed the private reflection seam those fixtures
use. A subsequent source change restores the original seven-argument constructor
and adds a separate video-pair overload. Its local focused video/isolation run
passed 33 tests with no skips. That later fix is **not part of r3**, and does not
change r3's failed-gates verdict.

The native TP CTest rows cover their compiled fixtures only; they do not clear
the separate expanded FP8/quantized checkpoint-shape failures or full-model
teacher-forced full-vocabulary comparisons. Trained draft heads, QSA above the
real checkpoint threshold, actual media/history, release cache reuse and final
performance gates remain separate requirements.
