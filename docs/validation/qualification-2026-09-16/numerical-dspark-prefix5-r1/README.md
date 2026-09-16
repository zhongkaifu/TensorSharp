# DSpark prefix5 observed-input diagnosis

The original CUDA confidence failure remains unresolved: 11 of 12 checks pass; the runner correctly returns exit 1. This one-GPU replay uses r3 native SHA256 `b6cf00377fca4154f8d318677e201044bf93adfa97ef26b0d6742ce1d4cbe751` on physical GPU 6 and unchanged upstream ggml `456172ec733a135778adcd32d00e576a58232e45`.

Long-double evaluation of the actual native target features produces stage 1/token 2/channel 35 prequant value 0.4208987751252852; the independent oracle features produce 0.4208983112106270. The BF16 midpoint is 0.4208984375. Both native and oracle correctly quantize their own different inputs, yielding FP8 cache values 0.4375 and0.40625 respectively. Re-evaluating the draft main projection, RMS normalization, KV projection and rotary transform with long-double arithmetic does not remove this difference. Maximum recorded target-feature difference is 4.172325e-6. The first cause therefore precedes the draft committed-feature projection; adjusting its quantizer or adding an epsilon would be unjustified.

The source diagnostic and selected recorded tensors are included. The FP64 full-reference script is retained as a diagnostic only; it does not replace the original F32 acceptance oracle or change any tolerance. Broader target arithmetic diagnosis and trained-checkpoint acceptance remain open.
