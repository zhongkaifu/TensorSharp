// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Isolated T8 scheduling experiment: include the unchanged TensorSharp source.
// Borrow only the small-token/two-stage geometry idea reviewed in Ignis e600;
// no original-Ignis code, raw layout, quantizer or output boundary is substituted.
#include "tensorsharp_q35_kernels.cu"

static_assert(TS_Q35_BM == 128 && TS_Q35_BK == 256);
static_assert(sizeof(TsQ35GemmSmem<8, 8, 1, 2>) == 43520);
#ifndef TS_Q35_HAS_GEMM_W4A4_8
TS_Q35_GEMM(ts_q35_gemm_w4a4_8, 8, 8, 1, 2, false, false)
#endif
