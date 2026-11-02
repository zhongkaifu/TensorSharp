CPU-only grammar mask / greedy argmax investigation

This independent console references only managed TensorSharp.Runtime. It never
loads weights, a backend, ggml or CUDA. Production sampling/streaming code is not
modified. It is an experiment, not a replacement for full TokenSampler.Sample.

From the repository root, export the frozen benchmark's schema without executing
its module or making a network request:

python3 - <<'PY'
from pathlib import Path
import ast,json
out=Path('artifacts/qwen38-regression/grammar-argmax-cpu')
out.mkdir(parents=True,exist_ok=True)
tree=ast.parse(Path('eng/qwen38-engine-bench.py').read_text())
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ORDER_SCHEMA' for t in node.targets):
        (out/'order-schema.json').write_text(json.dumps(ast.literal_eval(node.value))+'\n')
        break
else: raise RuntimeError('ORDER_SCHEMA not found')
PY

Offline build (an empty local feed plus existing NuGet cache; fails if required
managed packages are absent, never downloads a native dependency):

  mkdir -p artifacts/qwen38-regression/offline-empty-feed
  DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE=true dotnet restore eng/validation/Qwen38GrammarArgmaxBench/Qwen38GrammarArgmaxBench.csproj --source "$PWD/artifacts/qwen38-regression/offline-empty-feed" -p:NuGetAudit=false
  DOTNET_CLI_TELEMETRY_OPTOUT=1 dotnet build eng/validation/Qwen38GrammarArgmaxBench/Qwen38GrammarArgmaxBench.csproj -c Release --no-restore

Choose a fresh ignored output path. The tokenizer fixture below is the public
HF tokenizer JSON already available locally; no remote fetch is performed:

  DOTNET_TieredCompilation=0 DOTNET_TieredPGO=0 dotnet run --project eng/validation/Qwen38GrammarArgmaxBench/Qwen38GrammarArgmaxBench.csproj -c Release --no-build --no-restore -- --tokenizer docs/validation/qwen38-regression/cpu-preparation/tokenizer.json --schema artifacts/qwen38-regression/grammar-argmax-cpu/order-schema.json --out artifacts/qwen38-regression/grammar-argmax-cpu/local-run-new.json --rounds 7 --iterations 50

The real HF vocabulary is loaded with its declared added/special tokens, no
fabricated holes, and EOS im_end. This does not assert equality with a server's
GGUF tokenizer or padded model vocabulary. Six exact grammar states are exercised:
object opening, customer/product strings, integer quantity, patterned date and
complete object with EOS allowed. Cold vocabulary/mask work is separate from
warm sampling. Grammar.CurrentMask is looked up inside every timed alternative.

Baseline: actual Runtime TokenSampler.Sample with temperature0, no penalties,
no forced-thinking/first-token override, active nondead grammar. Comparison1:
standalone scalar mask+argmax fusion with immutable cached EOS word masks.
Comparison2: unchanged actual ApplyMask followed by a standalone SIMD argmax.
The tensor Core.Max implementation is not used: it returns a value rather than
the first global index and does not establish the sampler's NaN semantics.

Before timing, every alternative must match both token ID AND all post-mask
float bits against actual Runtime.Sample. Gates include real248077 and padded
248320 logits, small/short/padded/ragged arrays, EOS duplicate/negative/out-of-
range/>8 sets, ties, signed zero, infinities and leading/later/all NaNs including
signaling payloads. All-masked legacy behavior and unmasked logits tails remain
unchanged. No tolerance/skip or force option exists. Null/empty input and other
sampler configurations are outside this experiment, not claimed as passing.

SIMD uses ordered GreaterThan plus ConditionalSelect and carries original global
indices; first-index ties and leading-NaN behavior are retained. It does not use
Vector.Max. Leading-NaN early return is allowed only after the full original
mask has already run; fused masking never skips the remaining array writes.

Each timed iteration copies the same synthetic logits before the timer, warming
cache. ABBA compares actual baseline A against one prototype B at a time; whole
raw sample times are recorded, with no significance inference. Separate mask-only
and actual private scalar-argmax-only phases are diagnostic. JIT/tiering settings,
runtime/probe DLL SHA, tokenizer/schema SHA and CPU SIMD availability are recorded.

Local ARM64 measurements do not predict EPYC/AVX2, model quality, real-logit
distributions, GPU transfer latency or E2E behavior. In particular, fewer array
passes alone does not imply a win: scalar fusion may lose in broad-mask states.
Keep schema buffering/HTTP422 behavior and all sampling eligibility checks intact.
