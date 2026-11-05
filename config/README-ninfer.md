# NInfer Qwen3.8 shared configurations

The **same JSON file** works with `TensorSharp.Cli` and `TensorSharp.Server.Host`:

| Profile | Configured context |
|---|---:|
| [qwen3.8-27b-ninfer.json](qwen3.8-27b-ninfer.json) | 40,960; previously qualified engine profile, config parser tested |
| [qwen3.8-27b-ninfer-64k.json](qwen3.8-27b-ninfer-64k.json) | 65,536; actual CLI smoke and Host full45+30 tested |

Both select native `cuda`, original embedded full-head DFlash2, K7/pMin0 and
greedy sampling. They do not use `ggml_cuda` or an external draft model.
Both explicitly default `TS_SPEC_ADAPTIVE=0` to keep K7 drafting stable; this
does not assert an automatically qualified 64K governor policy.
The tokenizer, special-token metadata and chat template are in the `.ninfer`
artifact; no separate tokenizer option or file is needed.

Build with a toolkit supporting the target GPU, including the optional assets:

```sh
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release -p:CudaArch=compute_120
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -p:CudaArch=compute_120
```

Run from the repository root:

```sh
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/qwen3.8-27b-ninfer-64k.json --chat
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/qwen3.8-27b-ninfer-64k.json --host 127.0.0.1 --port 5088
```

Choose `qwen3.8-27b-ninfer.json` in either command for 40,960 capacity. For CLI
file input, replace `--chat` with `--input prompt.txt`; `--prompt` is not its
text-generation input option. Host-only options belong on the Host command,
not in a second application-specific configuration.

The model defaults to `<repo>/models/qwen3_8_27b_nvfp4full-v2.ninfer` and is
downloaded if absent: **19,406,942,468 bytes**, with a pinned revision and SHA256.
Set `TENSORSHARP_MODELS` to use another model directory, or use a CLI override
which skips the configured download entirely. For the existing validation VM:

```sh
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
  --config config/qwen3.8-27b-ninfer.json --chat \
  --model /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer
```

The reserved `environment` map sets process defaults after successful config
expansion, host preflight and downloads. Existing process variables win; later
config files win among defaults. Values must be strings. `${appBaseDirectory}`
is a reserved built-in naming the running application's output directory, so
one draft-module path finds each application's own `cuda_kernels` folder.
The profiles select `tensorsharp_q35_ninfer_draft.fatbin` directly; the variable
name still ends in `_PTX` but accepts fatbins. A PTX-only deployment can override
`TS_Q35_NINFER_DRAFT_PTX` with its real PTX path.
`appBaseDirectory` cannot be overridden through `variables`; `TS_Q35_NINFER_DRAFT_PTX` in the
process environment can explicitly select a nonstandard deployed asset.

40,960 is **not a file-format context limit**; the artifact declares 262,144.
The residual/norm, BF16 MMA and
GateUp fusion selectors admit exactly configured context **40,960 or 65,536**
on the qualified CC12.0/170-SM recipe. Their fixed T8/B1 mathematics do not use
context capacity; other prerequisites and shape checks still apply.

On RTX 5090, the 64K shared settings passed an actual CLI generation and Host
full45+30 evaluation: **38/45 text, 26/30 programs**, unchanged against R60 in
all 309 checks, 96 arithmetic values and 189 program statuses. All 123 serial
responses and 30 generated programs matched; 16 parallel contents changed
without score loss. Existing model mistakes remain. The 40K JSON was parser
tested; earlier engine validation does not imply that literal JSON was GPU run.

One 64K OFF/ON/ON/OFF study toggling the three matrix/norm optimizations together
measured code **406.49→442.42 token/s (+8.84%)**, HTTP total time **−8.10%**.
Responses, typed usage and draft work matched. A single seeded long diagnostic
repeated in all four arms used **64,416 input +144 output tokens, cache0** and
passed all ten needle/order checks each time. It is not general long reasoning
coverage. Complete 128K/250K campaigns for this optimized profile are **NOT_RUN**.

The final 64K preset differs from the GPU-tested `b652…` snapshot only in its
qualification comment; parsed settings are identical. Larger capacities need
more KV memory. Other hardware and arbitrary contexts are not qualified by
these results. Inspect startup selections/module SHAs and actual dispatch;
host submissions include capture, not individual GPU replays.

See the [native adapter guide](../TensorSharp.Backends.Cuda/native/ninfer/README.md)
for numerical contracts, automatic controls and finite validation results.
