# Raw-layout body projection gate

This module supplies the four concrete original NInfer body projections needed
by the separate runtime bridge: gate/up 34816x5120, down 5120x17408, output
5120x6144 and GQA input 14336x5120, each at physical T1 through T8. All32 static
images are mandatory. GDN input, BF16 exceptions, W8 heads and draft projections
are excluded. The current campaign builds the qualified N128 staged family
only; N64 metadata support does not establish native coverage or performance.

The self-contained builder verifies the four unchanged FlashInfer Python files
and license in the existing `flashinfer_w4a16_n64/upstream` source bundle, pinned
to `61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0`. It selects split1, four warps,
K128, two stages, no PDL, actual BF16 input and unrounded F32 output. It exports
standalone CUBINs, PTX, host launch IR and immutable source/toolchain/image
identities. TensorSharp serving uses only its CUDA driver loader.

The loader checks all32 geometries, five-pointer parameter ABIs, complete raw
code/scale spans, explicit family/policy, launch/shared reservations, actual
CUBIN hashes, one kernel per image and owning context. `ImageSetSha256` binds
the arithmetic policy and all32 actual loaded images in canonical shape/T order.
The caller must validate original finite scales and exact calibration, compact
already-BF16 activations, scatter original gate/GQA rows without narrowing, and
retain the existing residual/rounding/SwiGLU boundaries. The separate runtime
bridge owns those operations. Neither module nor component probe enables a
serving default or establishes quality equivalence to legacy accumulation.

The independent fixed component gate has four synthetic families x four shapes
x eight physical batches =128 cases. Two module lifetimes and three poisoned
outputs require256 unique lifecycle records and768 actual launches. It checks
every output value, exact F32 bits where mathematically proved, the unchanged
`gamma_(K+64)` FP64 bound elsewhere, guards, all input allocations, same-path
repeated bits and cross-lifetime bits. The 2,032 positive-finite E4M3/FP4 local
products must be exactly BF16. Extra active rows cover both original gate
halves, GQA Q/K/gate/V segment boundaries, 64/128-row boundaries and final rows.
The frozen independent oracle is reused without relaxed tolerances.

Only one shape/family payload and one case's device/CPU output/oracle allocations
are live at a time; results retain hashes and small metadata. Raw weights are
saved once per shape/family, and all complete captures are streamed directly to
the explicit fresh ignored output directory. Use the VM's evidence filesystem,
not its nearly-full temporary filesystem. Expect approximately1.2–1.4GB of
generated evidence. Write-once `completed-result.json` plus its SHA sidecar is
authoritative; `result.json` is progress/failure evidence.

```sh
python3 eng/validation/build-ninfer-flashinfer-body.py --verify-only
/root/venvs/vllm/bin/python eng/validation/build-ninfer-flashinfer-body.py --family n128 --out /fresh/artifacts/body-aot
dotnet build eng/validation/NinferFlashInferBodyProbe/NinferFlashInferBodyProbe.csproj
dotnet eng/validation/NinferFlashInferBodyProbe/bin/Debug/net10.0/NinferFlashInferBodyProbe.dll --self-test
dotnet eng/validation/NinferFlashInferBodyProbe/bin/Debug/net10.0/NinferFlashInferBodyProbe.dll --execute --manifest /body-aot/module.json --out /fresh/artifacts/body-components
```

CPU metadata tests and synthetic device checks are separate from real-model
projection inputs, end-to-end quality, cache lifecycle correctness and serving
performance. Small output-row shapes have fewer CTAs at split1 and must not be
assumed faster merely because the gate/up prototype was faster.
