# DSpark FP64 accumulation cost experiment

This isolated candidate is rejected for performance and is not integrated. Its corrected prefix 5 numerical result does not close the original DSpark confidence or full-vocabulary gates in the main binary.

Replacing small-width ordinary and indexed floating-point dot accumulation with FP64 costs approximately 1.79–5.99 times the original kernel latency on the A40. The benchmark retains 40 alternating, warmed CUDA-event samples per case, each measuring 20 kernel calls, and reports sampled output errors against independently accumulated CPU FP64 dots. The exact candidate source, build command, benchmark source, complete samples and binary hashes are retained here.

Other GPU campaigns were running on the VM. These measurements cover individual precision kernels on physical GPU 6, not quiet full-model latency, throughput or memory use. Prefix 5 correctness for the equivalent full-FP64 experiment was recorded separately; this archive records cost and cannot be treated as model qualification.

Dependency: unchanged upstream ggml `456172ec733a135778adcd32d00e576a58232e45`; baseline objects are from the recorded r3 build.
