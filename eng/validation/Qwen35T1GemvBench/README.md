# One-row projection microbenchmark (NInfer Qwen3.8 shapes)

Times the kernels that serve non-speculative decode on the NInfer route, one activation row
per projection, at the artifact's shapes:

| Shapes | Old kernel | New kernel |
|---|---|---|
| NVFP4 gate\|up [34816, 5120], down [5120, 17408] | `ts_ni_project_a16_legacy_exact_k*_t1` | `ts_ni_gemv_nvfp4_k*` (and `_swiglu`) |
| FP8 [16384, 5120], [14336, 5120], [5120, 6144], [34816, 5120] SwiGLU, [5120, 17408], head [248320, 5120] | `ts_q35_fp8_f32_t8_*` (8-row MMA tile) | `ts_q35_fp8_gemv_{b,c}_*` |
| add + RMSNorm, width 5120 | `ts_ni_add_norm` | `ts_ni_add_norm_5120` |

Weights are random (E4M3 codes and scales in a normal range) and rotated over four copies, so
no projection is served from the 96 MB L2. Each line reports microseconds, achieved TB/s
(codes plus scales) and the largest difference from the old kernel relative to the output's
largest value. The add+norm line counts values that differ in any bit; it must be 0.

```sh
nvcc -std=c++17 -O3 -gencode arch=compute_120,code=sm_120 -o bench bench.cu && ./bench
```

`compute_120` (not `sm_120a`) matches the product build, where CUDA emulates the E2M1
conversion. That emulation is why the NVFP4 kernel decodes code bits straight into F16.

RTX 5090, CUDA 13.2, driver 595.91 (2026-11-05):

```
nvfp4 n= 34816 k= 5120 legacy   102.9 us (0.97 TB/s)  new    55.9 us (1.79 TB/s)  maxrel 3.08e-07
nvfp4 n=  5120 k=17408 legacy    56.3 us (0.89 TB/s)  new    29.4 us (1.71 TB/s)  maxrel 4.70e-07
fp8 n= 16384 k= 5120 out1 mma  55.2 (1.52)  b  51.5 (1.63)  c  52.3 (1.60)
fp8 n= 14336 k= 5120 out1 mma  48.0 (1.53)  b  45.4 (1.62)  c  44.9 (1.64)
fp8 n=  5120 k= 6144 out0 mma  21.6 (1.45)  b  20.2 (1.56)  c  20.0 (1.57)
fp8 n= 34816 k= 5120 out2 mma 113.4 (1.57)  b 107.4 (1.66)  c 106.9 (1.67)
fp8 n=  5120 k=17408 out0 mma  60.8 (1.47)  b  54.8 (1.63)  c  54.6 (1.63)
fp8 n=248320 k= 5120 out1 mma 793.9 (1.60)  b 750.4 (1.69)  c 748.9 (1.70)
add_norm 5120: old 6.76 us  new 2.17 us  differing values 0
```

FP8 differences are up to one BF16 step for rounded outputs (out1/out2) and about 1e-6 for
plain F32 outputs (out0): the GEMV sums in a different order than the MMA tile. In the
server, non-speculative decode went from 53.3 to 73.2 tok/s (NInfer engine: 76.9).
