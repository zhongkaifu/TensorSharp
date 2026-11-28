# Graph op schema generator

The graph op protocol — op codes, operand lists, integer/float parameter slots, their defaults and bounds — is defined
once, in [`TensorSharp.Core/Graph/Schema/graph-ops.json`](../../TensorSharp.Core/Graph/Schema/graph-ops.json). This tool
generates its three consumers from it:

| Generated file | Consumer |
|---|---|
| `TensorSharp.Core/Graph/GraphOp.g.cs` | the IR: `GraphOp`, `GraphOpSchema` (table, `Check`/`Validate`, `Hash`, `Version`), `GraphOpParams` (slot constants and defaults), `GraphOpEmit` (typed emitters on `IGraphOpEmitter`) |
| `TensorSharp.Backends.GGML/GgmlGraphOp.g.cs` | the GGML managed bridge: `GgmlGraphOp` |
| `TensorSharp.GGML.Native/tsg_graph_ops.g.h` | the native bridge: `TsgGraphOp`, `signature()`/`check()`, `kHash`/`kVersion`, `slots::<op>::IP_*`/`FP_*`/`SRC_*` |

After editing the schema:

```bash
dotnet run --project eng/graph-schema            # regenerate
dotnet run --project eng/graph-schema -- --check # exit 1 when a generated file is stale (CI-friendly)
```

`InferenceWeb.Tests/GraphOpSchemaTests` fails while a committed file is stale, and the GGML backend refuses to initialize
when the native library was built from a different schema (`TSGgml_GraphSchemaHash` vs `GraphOpSchema.Hash`): rebuild
`TensorSharp.GGML.Native` after regenerating. The hash covers codes, names, slot structure, bounds and defaults, not the
docs; bump `version` in the schema when an op's meaning changes without a structural change.

Programs are checked against the schema while they are built (`GraphProgramBuilder.Op` and the native
`TSGgml_GraphOp`), so a missing operand or parameter is refused at graph construction, not when an executor reads past
its arrays. The runtime representation stays compact integer and float arrays; the typed emitters only shape them.
