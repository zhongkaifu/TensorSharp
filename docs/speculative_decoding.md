# Speculative Decoding in TensorSharp

Speculative decoding is a **speed** optimization and nothing else. A drafter
guesses the next few tokens, the trunk verifies them all in one batched forward,
and every emitted token is still drawn from a trunk row — so the output is what
plain decoding would have produced. A wrong guess costs a rollback, never a
wrong token.

This document describes how that is *built* in TensorSharp, and what you have to
write to add a new model or a new algorithm.

## The three layers

The design rests on one distinction:

```
   Model architecture      !=      Speculation algorithm      !=      Speculator weights
   ISpeculativeTarget              ISpeculator                        IDraftHead implementation
```

Conflating these is what makes speculative decoding hard to extend. They have
genuinely different lifetimes:

| Layer | What it is | Transfers between models? |
| --- | --- | --- |
| **Algorithm** | MTP / EAGLE / DFlash / DSpark / n-gram, as *code* | Yes — write once |
| **Runtime** | draft → verify → accept → rollback → commit | Yes — write once |
| **Weights** | the trained drafter in a checkpoint | **No** — bound to one target model |

A learned drafter reads the target's hidden states, borrows its tokenizer,
embedding and LM head, and is trained against its representation space. Even
when two models happen to share a hidden size, `h_Qwen != h_Gemma`. So the
weights are a model-specific artifact behind a thin adapter, exactly like a LoRA
checkpoint — while the framework and the runtime around them are not.

## The pieces

Everything lives in `TensorSharp.Runtime/Speculative/`.

### Layer 1 — the target model adapter (`ISpeculativeTarget`)

What the shared loop needs from the trunk, and nothing about drafting:

```csharp
void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows);
void SpecEnsureCapacity(int requiredSeqLen);
void SpecSnapshotRecurrentState();
void SpecRestoreRecurrentState();
void SpecRewindCache(int length);
int  CacheSeqLen { get; }
int  MaxContextLength { get; }
```

Two capabilities together make verification possible: a multi-row forward with
per-row logits (so a whole draft window is checked for roughly the cost of one
decode step, optionally tapping each row's hidden state), and a way to undo the
rejected tail.

`IBatchedSpeculativeTarget` adds the same thing over the batched paged path, so
the speculative trunk can run on the same kernels as the non-speculative batched
baseline and compose with prefix caching.

`ISpecTrunk` is the small seam that lets one loop serve both KV regimes:
`LinearSpecTrunk` (the model's live linear cache) and `BatchedSpecTrunk` (paged
KV + per-slot recurrent state, in `BatchExecutor`).

### Layer 2 — the algorithm (`ISpeculator`)

```csharp
int  Propose(in DraftContext ctx, List<int> draftOut);
void Commit(int[] tokens, float[] hRows, int startPos);
void Reset();
```

`Propose` guesses; `Commit` tells the speculator what actually landed in the
trunk, with the exact hidden states, so a learned drafter's KV cache and a
lookup drafter's corpus both track reality. `DraftContext` is one struct rather
than a parameter list precisely so a future algorithm that needs a new signal
gets a new field instead of a new overload everywhere.

Shipped implementations:

| Name | Class | Weights | Notes |
| --- | --- | --- | --- |
| `draft-head` | `DraftHeadSpeculator` | required | One token per pass, chaining its own hidden output: NextN/MTP (Qwen 3.6, GLM 5.2, GLM-5.3, Gemma 4's separate assistant GGUF). EAGLE-shaped heads fit here unchanged. |
| `block` | `BlockDraftSpeculator` | required | A whole block per pass with a confidence head: DeepSeek V4 DSpark, DFlash and DFlash2 (Muse-Glimmer, Qwen 3.8). |
| `ngram` | `NGramSpeculator` | **none** | Suffix matching over the sequence's own tokens (prompt-lookup decoding). Works on every model. |
| `auto` | — | — | Default: use whatever drafter the checkpoint carries. |

### Layer 3 — the weights (`IDraftHead`)

The model-specific adapter over a trained drafter:

```csharp
DraftHeadKind DraftHeadKind { get; }              // None | PerToken | Block
int  DraftBlockSize { get; }
void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut);
int  DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut);
void DraftCatchUp(int[] tokens, float[] hRows, int startPos);
```

A model with no drafter reports `DraftHeadKind.None` and still gets every
weight-free algorithm.

### The shared runtime

`SpeculativeExecution` is the one implementation of the protocol, for every
algorithm and every model:

1. **Draft** — `ISpeculator.Propose`. The loop does not know or care how.
2. **Verify** — the trunk forwards `[lastToken, d1..dK]` as ONE batch with
   per-row logits; the caller's sampler draws each row and drafts are accepted
   while the drawn token matches. Row *m*'s drawn token is the corrected (or
   bonus) token for free.
3. **Rollback** — on partial acceptance, restore the recurrent snapshot and
   re-advance over the kept prefix. Trunks whose verify already persisted usable
   KV (`SpecVerifyPersistsAcceptedKv`) skip the re-forward and just rewind the
   position — the dominant rollback cost on long contexts.
4. **Commit** — kept tokens go back to the speculator with their exact hidden
   states.

`SpeculationCostGovernor` sits alongside it, not inside it: speculation must
never make decoding slower, so the governor measures speculative against plain
steps at runtime and parks drafting while it is losing, re-probing with backoff.
It is a measurement *policy*, kept separate because it is the piece most likely
to be tuned or switched off per deployment.

`SpeculatorRegistry` maps a name to a factory. It is the only place that knows
which algorithms exist.

### Arming after a reused KV prefix

A sequence can begin from a KV prefix it never processed itself — the block-hash
prefix cache handed it over, or it is simply the next turn of a chat. The executor
used to refuse to arm speculation on any such sequence, because a learned
per-position draft head (NextN/MTP) chains its state token by token and a gap makes
every later proposal garbage. That is true of those heads, but it was applied to
every algorithm, and it cost the feature its whole point in ordinary use: from the
SECOND turn onward a Web UI conversation always adopts a prefix, so speculation
silently never armed and a DFlash drafter looked like it helped on turn one and did
nothing afterwards.

Which algorithms survive a gap is now the algorithm's own call —
`ISpeculator.CanArmAfterPrefixReuse`, default `false`. `BlockDraftSpeculator` and
`NGramSpeculator` opt in: n-gram mines the emitted token history, which is complete
whatever the KV cache did, and a block drafter reads its own sliding KV ring, which
refills from the freshly forwarded suffix and from every committed token, so an
adopted prefix costs it a shorter drafting context for a block or two and nothing
after that. Every draft is still verified by the trunk either way, so a stale
speculator can only cost throughput, never a wrong token. Measured in the server
chat path: 1.02x → 1.85x.

A per-token head can opt in too, through its weights adapter:
`IDraftHead.DraftHeadResumesAfterGap`. A NextN/MTP block with a KV cache of its
own (Qwen 3.6, GLM 5.2, GLM-5.3) keeps it false — a gap in what it replayed makes every
later proposal garbage. Gemma 4's assistant head keeps no state at all: every
draft step reads the trunk's donor KV and the hidden state it is handed, so it
drafts from any position and reports true. Without that, a Gemma 4 chat armed
its draft head on the first turn and never again.

**Not a reuse: the next chunk of the same prompt.** A prompt longer than one
prefill chunk arrives as several prefill steps, and every one of them satisfies
the executor's arming test (the trunk position always agrees with the scheduler
between chunks). The executor used to arm again on each chunk — an n-gram
drafter lost the tokens it had mined from chunk one, and from chunk two the arm
looked like a KV-prefix reuse at position 1024, so a per-token head was declined
for the rest of the request. With the phone's 1024-token chunk and a 5-7k-token
agent prompt that was every request TensorAgent made. The executor now keeps a
context that continues the sequence at the expected position
(`EngineSpec_PromptLongerThanOnePrefillChunk_StaysArmedAcrossTheChunks`).

**Arming after a plain prefill.** A turn that carries an image, audio or video
prefills on the plain path — only `Forward`'s inject hook can place the media
embeddings — and used to decode plainly to its last token, because nothing armed
once that prefill was done. A speculator that needs no hidden state now arms at
the first decode step, seeded with the tokens the trunk already holds
(`SpeculativeExecution.SeedCommitted`); a learned head is left alone, since no
trunk hidden state was captured for it to chain from
(`EngineSpec_MediaTurn_ArmsAHiddenFreeSpeculatorAfterThePlainPrefill`). The CLI's
single-shot path still declines `--spec` on a media turn.

**Arming on a per-request fused holder.** Every turn a real chat makes after its
first lives in a per-request fused holder — the clone of the shared-prefix
checkpoint that starts a new chat, the retained holder the previous turn left —
and the planner routes those to the per-sequence fused path, where the linear
speculative route was rejected ("sequence lives in a per-request fused cache").
A host that continues conversations from holders, TensorAgent, therefore never
speculated at all, whatever the setting said. The bound holder IS the model's
active cache (`BindSequenceCache` repoints the live arrays), so the linear trunk
runs on it unchanged; the fused path now keeps a speculative context per request
(`BatchExecutor.TrySpeculativeFusedDecode`), armed over the tokens the holder
already holds the way a plain prefill is, and disposed when the request leaves.

The linear path used to arm at the FIRST prefill chunk of every request and run
the whole prompt through the speculative context: the per-op speculative
forward, capturing a hidden row for every prompt token and catching the head up
over all of them. On a 1.2k-token first turn that cost 33% (E2B n-gram, 478
against 358 ms) to 36% (E4B with the draft head, 856 against 627 ms) of the
time to first token, and a request arriving behind it waited on that too. A
drafter that can be seeded from the trunk's tokens - n-gram, or a head that
resumes after a gap - now prefills plainly on the fused path and is armed by
the late-arming branch at the first decode step (`SeedCommitted`); only a head
that needs the prompt's hidden rows still prefills through the context.

One more gap sat on the linear path itself. When the executor armed at a prefill
chunk that started at a reused position (a follow-up turn continuing the live
cache), it never handed the reused tokens to the speculator: the n-gram
drafter's first commit then arrived at that position against an EMPTY corpus,
which it rightly treats as a gap it cannot account for, and it stayed silent for
the rest of the request. Every follow-up turn on the linear path drafted nothing
while the same turn on a holder drafted at 95%. The chunk arming now seeds the
execution with the reused prefix (`SeedCommitted`), the way the holder path and
the late-arming path already did; a learned head that cannot resume after a gap
declines there as before. `TS_NGRAM_DEBUG=1` prints the drafter's guard
decisions, which is how this was found.

## Adding a new speculation algorithm

Write the class and register it. No model, executor or scheduler code changes.

```csharp
public sealed class MedusaSpeculator : ISpeculator
{
    public string Name => "medusa";
    public int MaxDraftTokens { get; }
    public float MinDraftProb { get; set; }
    public float DefaultMinDraftProb => 0.6f;
    public bool NeedsHiddenState => true;
    public bool HandlesOwnPrefill => false;

    public int Propose(in DraftContext ctx, List<int> draftOut) { /* ... */ }
    public void Commit(int[] tokens, float[] hRows, int startPos) { /* ... */ }
    public void Reset() { }
    public void Dispose() { }
}

SpeculatorRegistry.Register("medusa",
    (target, options) => target is IDraftHead h && h.DraftHeadKind == DraftHeadKind.PerToken
        ? new MedusaSpeculator(h, target.Config.VocabSize, target.SpecFeatureSize, options.MaxDraftTokens)
        : null,
    requiresDraftHead: true);
```

Return `null` from the factory when the algorithm cannot serve that model; the
registry turns it into an operator-facing decline reason. `requiresDraftHead`
lets the execution planner explain "no draft head" up front instead of routing a
request onto a path that will bail.

Correctness is **not** the algorithm's responsibility. Whatever it proposes,
verification emits only tokens drawn from a trunk row. A speculator is free to
be wrong; it must not be slow.

## Adding a new model

Implement `ISpeculativeTarget` on the model — the multi-row forward plus the
rollback trio — and it can immediately be sped up by every weight-free
algorithm. If the checkpoint also ships a drafter, implement `IDraftHead` on the
same class (there is an `ISpeculativeModel` alias for the pair) and report the
matching `DraftHeadKind`.

One caveat worth stating: some models' `SpecForward` is not drafter-independent
— Muse-Glimmer and DeepSeek V4 share the fused verify kernel with their drafter
and refuse to run without it. Those report `SpeculationProfitable` as false when
no drafter is loaded, so weight-free speculation is declined rather than
crashed. Qwen 3.5/3.6, GLM 5.2, GLM-5.3 and Gemma 4 have drafter-independent trunks and
accept `--spec-type ngram` on any checkpoint. (Gemma 4 used to be gated on its
assistant GGUF too; the gate was an artifact — its multi-row verify is the same
fused whole-model kernel its prefill runs, and the hidden-state capture is only
filled when a speculator asks for it. TensorAgent ships the assistant GGUF as an
optional download, so this is what makes speculation reachable there at all.)

## Operator surface

```
--spec | --no-spec              enable/disable speculative decoding
--spec-type <name>              auto (default) | draft-head | block | ngram
--spec-draft <N>                max tokens drafted per step (1-64, default 8)
--spec-pmin <f>                 confidence gate (default: per algorithm)
--draft-model <path>            a drafter that ships as its own GGUF: Gemma 4's
                                draft head, or a block drafter resident before
                                the layer split
```

The historical `--mtp-spec`, `--mtp-draft`, `--mtp-pmin` and `--mtp-draft-model`
spellings (and the old `--spec-draft-model` alias) have been removed: each fails
with an error naming its replacement, never a silent ignore, because the CLI's
argument switch drops unknown flags and "speculation quietly off" is exactly the
failure that would produce. Environment variables are published under both
`TS_SPEC_*` and `TS_MTP_*`, and that is not merely for compatibility: the glm-dsa
**native** loader reads `TS_MTP_SPEC` and `TS_MTP_DRAFT` from C++ while the model
is loading — it decides whether to page a whole extra 256-expert decoder layer
into VRAM, and sizes its graph cache — so those names are a cross-language
contract.

`--spec-pmin` means something different per algorithm, which is why each brings
its own default rather than sharing one: `0.15` for a per-token head (top-1
probability over its top-10 logits), `0.35` for a block drafter (the CUMULATIVE
prefix probability, so the same number is far stricter), `0` for n-gram (where it
scales the required match length instead).

## DFlash and DFlash2

A **DFlash** drafter is a small block-diffusion model that ships as its own GGUF
(`general.architecture = dflash`) and is bound to one target. It reads the
target's own residuals rather than only its tokens, and it proposes the whole
speculative window in ONE forward pass instead of one token at a time:

```
PASS A  encoder      feat = concat(target residual entering dflash.target_layers)
                     g    = rmsnorm(fc @ feat, enc.output_norm)
PASS B  KV inject    K = rope_neox(headnorm(attn_k @ g)) ; V = attn_v @ g
                     ring[pos % ringRows] <- K, V        (no Q, no attention, no FFN)
PASS C  block draft  ids = [anchor, MASK x (block_size-1)]
                     -> draft blocks -> the TARGET's LM head -> block_size-1 drafts
```

The drafter owns a small sliding-window KV ring of its own, sized from
`dflash.attention.sliding_window`; the target's KV cache is untouched. Everything
the drafter needs beyond its own blocks - the token embedding and the LM head -
is borrowed from the target, so the file is ~1-3 GB against a 27-30B trunk.

**DFlash2** is the same backbone with two additions, both keyed off the GGUF, so
one code path serves both generations:

* **A grouped dynamic depthwise convolution** around every attention and every
  FFN sublayer (`dflash.conv_kernel_size`, `dflash.conv_group_size`). One
  projection of the sublayer's INPUT produces both the filter applied to that
  input and the filter applied to the sublayer's OUTPUT. Tap *t* of channel *c*
  at block position *r* is `base[t][c] + delta[r][t][c / group_size]` - static
  per channel, dynamic per group - multiplying `x[r-t][c]`, and masked to zero
  for `r < t` so the filter never reaches across a block boundary. It is what
  gives a block-diffusion draft a local left-to-right signal without a second
  forward pass.

* **A candidate selector** (`dflash.selector_rank`, `dflash.selector_top_k`).
  Plain DFlash takes each block position's argmax over the vocabulary
  INDEPENDENTLY - exactly the weakness of block diffusion, since position *i+1*
  is chosen without knowing what *i* chose. The selector keeps the top-K
  candidates per position and scores every (predecessor, candidate) pair through
  two low-rank `[vocab, r]` codebooks:

  ```
  score[e][p][c] = unary[e][c] + < A[pred[e][p]] * (P h_e) , B[cand[e][c]] >
  ```

  `A`/`B` are `selector_predecessor`/`selector_successor`, `P` is
  `selector_hidden`, `pred[0]` is the verified anchor token and `pred[e]` is
  `cand[e-1]`. The block is then read off as a greedy walk through that lattice:
  one small matmul per position, no extra draft forward.

  `unary` is the target LM head's logit for that candidate **after the target's
  own logit transform** (`dflash.logit_scale`, `dflash.final_logit_softcapping`),
  which is why those keys exist on a DFlash2 file at all. Plain DFlash takes an
  argmax and is invariant to both; the lattice ADDS the unary term to a
  transition score, so an untransformed unary is simply the wrong size and
  swamps the transition it is meant to compete with. Skipping it on the
  Muse-Glimmer drafter (scale 0.196, softcap 20) cost more than half the
  acceptance rate.

Both extensions are no-ops when their keys are absent, so a first-generation
DFlash file runs through the same code unchanged. `TS_DFLASH_SELECTOR=0` and
`TS_DFLASH_CONV=0` switch one off for attribution; neither is a supported way to
run a model, since the weights were trained with both.

### Where it runs

Both passes are one fused GGML graph each (`ggml_ops_dflash.cpp`,
`TSGgml_DFlashInject` / `TSGgml_DFlashDraftBlock`) on CUDA, Vulkan and Metal,
with a persistent graph that ggml-cuda can capture and replay; the per-op
managed drafter is the fallback and the reference the fused path is checked
against. `TS_DFLASH_FUSED=0` forces it.

The selector's lattice comes back to the host as `k + k*k*(gamma-1)` floats
(~7 KB) rather than the `[vocab, block]` block a naive readback would move
(12.9 MB), and the walk itself - inherently sequential, tiny - runs on the host.

### Attaching one

`--draft-model <path>` (or `TS_QWEN35_DFLASH` /
`TS_MUSE_GLIMMER_DFLASH`). The file's `general.architecture` decides what it is,
not its name. A target that already carries a NextN/MTP block (Qwen 3.8 does)
uses the DFlash drafter instead when one is attached: they consume different
hidden rows and drive different speculators, and the operator named the file
explicitly.

### What the target has to provide

Only the residual tap. A DFlash target implements `SpecForward` so that, per
row, it also writes the concatenated residuals ENTERING each layer in
`dflash.target_layers` - `SpecFeatureSize` wide instead of one hidden. Both
shipped targets do it inside their fused whole-model kernel (a `ggml_cpy` per
tapped layer), so speculation does not force the op-by-op loop.

### What to expect

Measured on one RTX 3080 Laptop (16 GB), greedy, best of two runs. Two prompts,
because acceptance - and therefore everything - depends entirely on how
predictable the continuation is: a free-form "explain how a GPU does a matmul"
(prose) and a "list the first 20 primes" (factual).

| target | drafter | prose tok/s | factual tok/s |
| --- | --- | ---: | ---: |
| Muse-Glimmer 30B IQ2_XXS | none | 18.7 | - |
| Muse-Glimmer 30B IQ2_XXS | DFlash (1.6 GB) | 25.4 (1.36x) | - |
| Muse-Glimmer 30B IQ2_XXS | DFlash2 Q4_K_M (1.6 GB) | 23.0 (1.23x) | - |
| Muse-Glimmer 30B IQ2_XXS | DFlash2 Q8_0 (3.0 GB) | 14.1 (0.75x) | - |
| Qwen 3.8 27B IQ3_XXS | none | 17.9 | 17.1 |
| Qwen 3.8 27B IQ3_XXS | NextN/MTP | 20.4 (1.14x) | 30.1 (1.76x) |
| Qwen 3.8 27B IQ3_XXS | DFlash2 Q4_K_M | 15.3 (0.85x) | 25.8 (1.51x) |
| Qwen 3.8 27B IQ3_XXS | DFlash2 Q4_K_M, `--spec-draft 7` | 9.8 (0.55x) | 23.5 (1.37x) |

The four Qwen rows are one uninterrupted sweep, so they are comparable to each
other; the Muse-Glimmer rows are from a separate one and are not comparable to
them in absolute terms.

Treat the absolute numbers as indicative, not exact. On this laptop card a plain
decode - which does identical work per token whatever the prompt - measured 18.7
and 16.5 tok/s in two back-to-back runs of the same binary. Anything under about
15% apart on a single run is noise here; the comparisons below that matter were
all made as paired runs, alternating the configurations inside one batch.

That caveat is not theoretical: an earlier revision of this page reported DFlash2
on the prose prompt at 20.9 tok/s (1.14x), and it does not reproduce. Repeated
paired runs put it at 0.85-0.96x - break-even at best on free-form prose - while
the plain baseline measured beside them barely moved. The factual rows and the
MTP rows did reproduce. Believe the ratios, re-measure before believing a
single-run figure, and do not compare a number here against one taken on another
day.

### Against llama.cpp

llama.cpp b10630 on the same files and card: Muse-Glimmer plain 19.7 / DFlash
22.0. It cannot load a DFlash2 drafter at all - it rejects the file with "wrong
number of tensors; expected 81, got 58", the 23 convolution and selector tensors
it has no code for - so on DFlash2 there is nothing to compare against.

On MTP there is, and getting it right took two corrections. A first pass ran the
two engines on the same prompt without noticing that llama.cpp turns thinking
mode ON by default for this checkpoint and TensorSharp does not, so they were
answering with different continuations; since acceptance is a property of the
continuation, that measured the text rather than the engine. The numbers below
are a true like-for-like: same prompt, thinking disabled on both
(`chat_template_kwargs: {"enable_thinking": false}`), greedy, 256 tokens, draft
window 3.

| | tokens/accept call | acceptance | tok/s | ms/step |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp `draft-mtp` | 3.63 | 0.885 | 39.4 | 92.1 |
| TensorSharp `--spec` | 3.67 | 0.932 | 33.9 | 97.4 |

**Drafting is at parity or better** - TensorSharp gets slightly more tokens per
verify call than llama.cpp does. The gap is entirely per-step cost, and
TensorSharp's own phase counters locate it: a verify is 77 ms (llama.cpp's works
out to about the same), and the draft calls are 20 ms against roughly 13.

One caution about llama.cpp as a yardstick: its eval time reproduces to within
0.04% run to run on this card (2886.84 ms and 2885.74 ms on two identical
requests), where TensorSharp swings by several percent. The variance is
TensorSharp's, not the machine's.

#### What was eliminated

llama.cpp runs its MTP block ONCE over `n_accepted + 1` rows, folding the
catch-up over the accepted tokens and the first draft step into a single call.
TensorSharp ran a catch-up and then a separate first `DraftStep`, and on a head
whose per-call cost is mostly fixed that extra call was the largest single
difference. It now folds too (`SupportsFusedCatchUpStep` /
`DraftCatchUpAndStep`, `TS_MTP_FOLD_CATCHUP=0` to revert): `catchUpMs` 191 -> 0,
worth +4.0% at 256 tokens and +5.3% on prose, with byte-identical output and
unchanged acceptance.

#### What is left, measured

Three things were checked and are NOT the problem, which is worth recording
because each looks like an obvious suspect:

- **CUDA-graph capture of the verify.** `TS_GGML_LOG_DEBUG=1` surfaces ggml's
  "CUDA graph warmup complete"/"reset" lines. Capture does churn (the persist
  cache evicts across draft shapes), but raising
  `TS_Q35_VERIFY_CACHE_BUDGET_MB` from 1536 to 3072 halves the resets and
  changes throughput not at all.
- **The MTP draft graph not persisting.** `TS_Q35_MTP_DRAFT_PERSIST=1` moves
  `draftMs` by less than the run-to-run noise.
- **The confidence gate.** llama.cpp does not gate at all; dropping `--spec-pmin`
  to 0.05 is a wash on both prompts, because the steps it declines genuinely
  would have drafted badly.

What IS left is the per-call overhead of the MTP block. Instrumenting the two
halves over a 256-token run: the C# input projection (`MtpProjectInput` -
embedding, two RMS norms, a concat and `eh_proj`) costs 462 ms against the fused
block kernel's 804 ms, over 208 calls. That is 2.2 ms of every 6.1 ms draft
call, and **6.4% of the whole run**, spent on about six separate device op
launches. Caching its scratch tensors changes nothing (the allocator already
pools), so the cost is the launches themselves: on CUDA every op synchronises,
because the lazy-sync path (`TS_GGML_ASYNC_COMPUTE`) is Metal-only - it relies
on Metal's zero-copy host mapping. Folding the projection into the fused MTP
graph, so the whole draft step is one graph, is the next concrete step.

Three things in that table are worth reading carefully.

**The drafter's SIZE is a first-order performance variable on a card with no
headroom.** The same DFlash2 drafter at Q8_0 (3.0 GB) instead of Q4_K_M (1.6 GB)
turns a 1.23x win into a 0.75x loss - not because it drafts worse (its
acceptance is identical) but because the extra 1.4 GB pushes the trunk into
WDDM paging and the trunk's own verify slows from 78 ms to 128 ms. Match the
drafter quant to the headroom, not to the best available fidelity.

**The window is a workload choice, and the default is the conservative one.**
Qwen 3.8 defaults to 3 (see `SpecPreferredDraftWindow`). Widening it to 7 costs
9% on the factual prompt and 36% on prose, because a wider window buys verify
rows that get rejected AND makes the recurrent-state snapshots below
proportionally larger - ~150 MB per slot here, which on a card with no headroom
is its own second penalty. `--spec-draft N` overrides it. (An earlier revision
claimed a window of 7 WON by 10% on the factual prompt; that was measured before
the state stopped round-tripping, when a wider window amortised a fixed per-step
transfer that no longer exists.)

**What the drafter proposes is only half the story on a recurrent trunk** - the
other half is what a REJECTION costs, which is the next section.

## Rejection on a recurrent trunk

Qwen 3.5/3.6/3.8 are hybrids: 48 of Qwen 3.8's 64 layers are GatedDeltaNet, and
GDN carries a recurrent state that a KV cache's "drop the rejected tail" does not
apply to. Rolling a partially-rejected verify back used to mean restoring a
pre-verify copy of that state and re-forwarding the accepted prefix through the
entire trunk - a second whole-model forward - because the state after row *m*
simply did not exist anywhere. On top of that the state (151 MB for this model)
crossed PCIe twice per step: uploaded into the verify graph, downloaded again
after it. Speculation therefore cost MORE than the plain decode it was meant to
beat: 15.5 tok/s against 18.3 for DFlash2, 15.7 against 18.3 for MTP.

Three changes, all in the fused verify kernel and its Qwen 3.5 caller, removed
that (`ggml_ops_qwen35_verify.cpp`, `Qwen35Model.GatedDeltaNet.cs`):

1. **The verify keeps one recurrent-state snapshot per row.**
   `ggml_gated_delta_net` already takes a snapshot count and emits the last K
   per-token states; the conv state after row *m* is a window of a tensor the
   graph already builds. The state a rollback wants is therefore never
   recomputed - it is slot `N-1-accepted`.

2. **A snapshot is committed into the live state on the DEVICE.** Every cached
   verify graph binds its `*_state_in` from one shared device buffer, so writing
   a slot into it is visible to the next verify whatever shape it runs at. The
   state stops round-tripping: the next verify skips its upload, this one skips
   its download.

3. **The pre-verify snapshot becomes free.** A verify only READS the live
   slices - it writes its results to `*_state_out` and the snapshot slots - so
   the slices ARE the pre-verify state until a commit overwrites them, and a
   commit only happens after the rollback decision. `SpecSnapshotRecurrentState`
   copies nothing.

4. **The single-row steps stop round-tripping too.** A speculative session is
   not all verifies: when the drafter declines to propose, the step falls
   through to an ordinary one-row forward, and those ran the old download.
   Each one broke the device-state chain - 151 MB down, and the *next* verify
   had to upload it again - which on an MTP run (46 such steps out of 125) was
   most of what was left. A one-row step's post-window state is simply the
   `*_state_out` slices and nothing decides anything about it later, so the
   kernel now defers it as well and the caller commits slot -1 immediately:
   one device-to-device copy instead of 302 MB across PCIe.

The measured effect on Qwen3.8-27B, DFlash2, 256 prose tokens: `rollbackMs`
3604 -> 0, `snapshotMs` 919 -> 69, and 15.5 -> 20.9 tok/s. On the factual prompt
24.3 -> 31.7. Deferring the one-row steps (4) is worth a further 5-20% on top,
paired-run: DFlash2 factual 22.0 -> 27.1, MTP prose 19.1 -> 21.4.

Output is unchanged - in fact it is *more* exactly unchanged than before.
Committing on the device is a raw copy of the tensor the graph produced, where
the host round trip went through the state's unpack-and-repack; on the factual
prompt the device path reproduces plain decoding byte for byte while the host
path drifted in the last few tokens.

`TS_Q35_VERIFY_SNAPSHOTS=0` restores the old path entirely, and
`TS_Q35_VERIFY_DEFER_STATE=0` keeps the snapshots but restores the download, so
the two halves can be measured apart. Either is also what a shape the kernel
will not persist falls back to, automatically. The cost of the snapshots is
VRAM: the GDN op's output grows by one state per slot, ~150 MB per slot for this
model across all 48 recurrent layers, which is the other reason the default
window is 3 rather than 8.

## Sliding-window caches and rollback

A verify writes every row's K/V at its true position. Gemma 4's local (SWA)
layers keep exactly one window of positions in a circular cache — slot =
position % 512 — so once the context has wrapped, row `p+i` of a verify lands on
the slot that held position `p+i-512`. The token decoded right after a rollback
still attends to that position whenever two or more rows were rejected (its
window is `[q-511, q]` with `q = p+m+1`, and `p+i-512 >= q-511` for every
`i >= m+2`). Rewinding the position counter cannot bring those rows back, and
neither can the kept-prefix re-forward: it rewrites the accepted rows, not the
evicted ones. `Gemma4SwaRollbackExactnessTests` measures the damage on E2B: under
the window the first decode after a rollback agrees with plain decoding to 2e-3;
at a 900-token context it was off by 2.4-3.0 logits on a scale of 20, and greedy
output diverged within a couple of dozen tokens.

The trunk now keeps what a verify is about to evict — the rows of every
non-shared SWA layer whose slots the verify overwrites, ~16 KB per layer — and
puts the rejected ones back in `SpecOnVerifyAccepted`, before any rollback
decision. Only those byte ranges move between the host mirror and the device
copy (`TSGgml_SyncHostBufferRanges` / `TSGgml_UploadHostBufferRanges`), never
the whole cache. A trunk with a linear cache (every global layer, Qwen 3.5's
attention layers) needs none of this: a rewound position simply overwrites the
rejected rows later.

## How many rows to verify

ggml's small-batch matmul kernels — ggml-metal's `mul_mv_ext`, ggml-cuda's
`mul_mat_vec_q` — serve 2 to 8 rows. A verify of 9 rows tips every matmul in
the graph onto the large-batch path, and that is what the default window of 8
produced. Measured on Gemma 4 E4B Q8_0 (M5 Pro, Metal, 1.1k-token context,
greedy, 160 tokens, plain 46 tok/s):

| window | verify rows | verify ms | draft head tok/s | n-gram tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 9 | 89 | 69 | 49 |
| 7 | 8 | 51 | **92** | 63 |
| 5 | 6 | 44 | 89 | **70** |
| 3 | 4 | 38 | 80 | 66 |

Gemma 4 therefore prefers a window of 7 on the ggml backends
(`SpecPreferredDraftWindow`), the way Qwen 3.5/3.8 prefer 3 for their recurrent
state, and the preference now applies to every algorithm — n-gram used to take
the raw option and verified 9 rows on trunks that had asked for 3 or 7. An
explicit `--spec-draft` still wins.

Two more things were hiding in the 8-row verify, both found with the phase
counters (`TS_GMTP_PROFILE=1`) on the real TensorAgent host path:

- **The per-op tail.** After the fused kernel the trunk ran the output norm, a
  262K-vocab LM-head matmul over the 8 rows and the tanh softcap as separate ops
  with host round-trips between them: 13 ms of a 58 ms E4B verify, 11 of 34 on
  E2B. The verify kernel now folds all three into its graph, the way the decode
  kernel already did for a single row (`TSGgml_Gemma4ModelVerify`'s trailing
  `logits_data / lm_head / final_norm / logit_softcap` arguments; the model
  passes them whenever it wants every row's logits and the output weight is
  quantized, `CanFoldLmHead`); the rows come back post-norm, which is what the
  draft head consumes, and the tail costs about 3 ms inside the graph. E4B: 58
  to 48 ms per verify. `TS_GMTP_NO_FOLD_HEAD=1` restores the tail for an A/B.
- **IQ4_XS had no small-batch kernel on Metal.** ggml-metal's `mul_mv_ext`
  covers Q4_0/Q5_0/Q8_0/IQ4_NL for 2..8 rows and the K-quants for 4..8, but not
  IQ4_XS, which is what 234 of the E4B catalog model's tensors use (unsloth's
  gemma-4-E4B-it-IQ4_XS); an 8-row verify on it fell to the per-row `mul_mv`
  path and cost 4.5x a single-row decode. An earlier local ggml patch added
  the missing dispatch and reduced the trunk kernel from 53 to 42 ms. That
  patch has been removed: dependency fetching now consumes unchanged upstream
  ggml. TensorSharp instead uses a shorter automatic draft window for Metal
  trunks with IQ4_XS matrices and gathers PLE inside the existing fused verify
  graph. This avoids spending matrix work on long rejected suffixes and removes
  the separate PLE device/host round trips. Explicit draft-window settings still
  take precedence; other quantization formats keep their existing defaults.
  See [unpatched ggml validation](perf/ggml-without-patches.md) for the current
  measurements and reproducible checks.

The earlier patched implementation was measured by varying the window on the same host path (E4B
IQ4_XS, draft head, folded head, 5k-token context; plain step 15 ms):

| verify rows | verify ms | quote turn tok/s |
| ---: | ---: | ---: |
| 2 | 26 | 69 |
| 4 | 32 | 94 |
| 8 | 49 | 101 |

A line through those is 16 ms fixed plus 3.7 ms per row. The fixed part is the
verify graph itself: 2,136 nodes (51 per layer) built and encoded on every call,
where the decode kernel captures its graph once and replays it with input
refreshes (`G4DecodeCache`, rebuilt only when a window crosses a 256-token
stride). The slope is the small-batch kernel's design: `mul_mv_ext` handles
`r1ptg` = 4 rows per threadgroup and dispatches `ne11 / r1ptg` of them, so 8
rows read the weights twice. Persisting the verify graph per row count is the
next step on this path and is not done; it would take the 8-row verify to
roughly 35 ms. Until then, on E4B a verify is 3x a plain step, break-even is
about 30% acceptance, and the governor is what keeps prose from paying for it.

A parked speculator runs nothing but plain steps, and those used to go through a
one-row `SpecForward` — the fused decode kernel followed by that same per-op
tail — 14% (E4B) to 19% (E2B) slower than the ordinary decode while drafting
nothing. A speculator that needs no hidden state (n-gram) now takes the model's
own `Forward` for its plain steps on trunks that declare the two paths leave
identical state behind (`ISpeculativeTarget.SpecPlainStepUsesForward`; Gemma 4).

## Stops inside a verify window

A chat prompt contains the turn boundary several times, so an n-gram drafter
proposes exactly that continuation after an end-of-turn token, and the trunk
agrees with it. The accept loop used to take those rows: the engine then ended
the sequence at the stop and trimmed the token list, but the accepted rows past
it stayed in the cache — a retained holder was longer than the tokens it was
recorded as holding, and on a sliding-window ring those rows had evicted
positions the next turn still attends to. The executor's draw hook now returns
no token once a stop (end-of-turn, or the repetition guard) has been accepted,
so the loop ends there and the rows past it are rejected rows, which the ring
backup restores; a window never starts at a stop token either
(`EngineSpec_EosInsideAcceptedWindow_StopsAndTruncatesTrailingDrafts` checks
the trunk holds exactly the prompt and the emitted tokens).

## What the governor measures

The cost governor's verdict is about the (model, drafter, backend) triple, so
the executor now shares ONE governor across every request it arms: a chat's
short turns no longer each pay a fresh probe round to rediscover that drafting
loses on prose (it still re-probes with backoff). And while a learned head that
can resume after a gap is parked, its plain steps take the model's own decode
(the carry is captured again by the first step after the park) — before, the
parked baseline was a one-row speculative forward with a per-op head, dearer
than the real plain step, and the governor kept calling speculation a win
against that inflated baseline: E4B-IQ4_XS on the host benchmark ran prose at
23-40 tok/s under it, against 65 plain. A request that cannot decode two tokens
(`maxTokens: 1`, the warm-up) is never armed.

The probe itself is paid in speculative steps, which on a losing pairing are the
dear ones (E4B-IQ4_XS with its draft head spends about four plain steps per
speculative one, so an eight-step re-probe every 64 tokens kept prose 30-50%
slower while nominally parked). After a losing verdict a re-probe is three steps
and the park backs off to 256 tokens; a winning verdict restores the full probe.

A winning verdict is not a blind hold either. Every turn starts with the same
thinking preamble ("The user is asking me to write three short paragraphs..."),
which a draft head predicts almost perfectly, so the first probe of a prose turn
wins honestly and the free prose behind it then loses at 12-17% acceptance: held
for 64 steps that ran E4B-IQ4_XS prose at 30-34 tok/s against 65 plain in two of
three runs (61-69 verify steps of ~80). While a win holds, the governor keeps
timing its speculative steps against the plain baseline the round measured and
parks as soon as the last eight of them average a loss, so a wrong win now costs
about eight steps. The hold is still re-measured every 64 steps (three plain
steps, since the speculative samples are productive) to refresh the baseline.

The governor is shared by every request an executor runs, so a park earned on
one turn's text would carry into the next: a quote turn after a prose turn
reached 99% acceptance and still ran 122 of its 135 steps plain under a 128-step
park. A new request caps whatever park is left at the first interval and
forgets the backoff (one loss is remembered, so the re-probe stays the short
one): the new turn re-probes within its first 32 steps, and a re-probe that
loses parks 32, not 256 - with the backoff kept, one noisy three-step re-probe
parked the remaining 200 steps of a quote turn that runs 85 tok/s against 65.
The first park is 32 rather than 16 because on a losing turn each re-probe is
three dear speculative steps, and 16/32/64/128 fitted four of them into a
160-token prose turn where 32/64/128 fits two.

The per-turn stats line reports what the governor did:
`governor plain=15.3ms/tok spec=29.1ms/tok wins=0 losses=2 parked=96`.

A probe also ends early: once four measured speculative steps in a row have
accepted nothing, the verdict is a loss and the remaining probe steps - each a
verify costing 2-4x a plain step for one token - are not paid. An n-gram drafter
on free prose accepts nothing for whole turns (30 drafted, 0 accepted over 10
verifies on Qwen 3.5-9B).

Two accounting rules matter for those verdicts. A step whose draft HEAD ran and
proposed nothing (under its confidence gate) is recorded as a SPECULATIVE sample
of one token: the head pass is in its cost, and charging it to the plain side
inflated the baseline every verdict is measured against; worse, a head that
stayed under its gate never filled the probe's speculative quota, so the round
never closed and the head ran unparked on every step. A matchless n-gram
lookup is recorded the same way even though it paid no head pass: its step ran
on the speculative path's plain step, which on a trunk whose state families
differ (Qwen 3.5) is the dearer one, and that is precisely the cost a parked
step avoids; recording it as plain instead inflated the baseline and let n-gram
run through prose unparked (Qwen prose 47 to 38 tok/s). The consequence that
four matchless steps park an n-gram drafter is intended: it re-probes within
32 steps. And the plain baseline is taken with the
model's own decode step while the speculator is parked or calibrating, never
with a one-row speculative forward: on Qwen 3.5 that costs a state-family
switch each way per round (the verify family's state drained to the host, the
decode graph re-seeded, and back), which is the price of a baseline that is
not 25% inflated - an inflated one made prose look like a win.

Two more rules protect the trunk. The sampled token itself can complete a
repetition loop; the engine then stops at it and truncates every accepted draft
from the sequence, which a bound holder that cannot be truncated (Qwen 3.5)
would keep, leaving the next turn a cache longer than its tokens. The executor
therefore runs the guard on the sampled token before drafting and verifies
nothing past a token it is about to stop at. And Qwen's pre-verify recurrent
snapshot settles the fused decode's device-resident state into the host mirrors
first: a parked run leaves the state in the decode graph's slot, the executor
snapshots before the verify forward that would sync it, and a rollback from a
stale snapshot (the host-mode verify, the per-op fallback) would continue from
the pre-park state.

Greedy verification is exact only up to the kernels' arithmetic: the 8-row
verify kernel and the 1-row decode kernel accumulate in different orders, so a
token that is a near-tie in the logits can come out differently depending on
which kernel produced it - and which one did depends on the governor's timing.
The stream-equality benchmarks pass on their prompts; on the host benchmark one
run in ten took a different (still well-formed) greedy path at such a tie.

A holder can only be speculated on by a trunk that forwards on the BOUND cache
(`ISpeculativeTarget.SpecTrunkFollowsBoundCache`; Gemma 4 and Qwen 3.5 both
declare it, the default is false and the executor warns once for a model that
does not). Qwen 3.5 used to lose the request at position 0 when tried: not
because its state went to the wrong cache - every field the speculative trunk
touches is what `BindSequenceCache` swapped in, and a holder switch drains the
verify graph's device-live recurrent state into the outgoing holder first - but
because entering its speculative session flipped `SupportsPerSequenceFusedForward`
off, so the planner re-routed the holder-resident sequence to the linear path on
the next step. The capability no longer depends on the session; entering the
session is a one-time transition (it used to hard-drop every holder's Metal
decode graph on every speculative step) and the fused decode leaves the session
itself, draining the verify family's state, when it is next asked to run.

Qwen's plain steps have a second subtlety. Its fused decode and its verify graph
keep the recurrent state in different device families, and switching costs a
drain and a re-seed each way (~50 MB per direction on the 9B, ~150 MB on the 27B). A one-row pass through
the verify family costs 25.6 ms against 20.4 for the fused decode, so a parked
speculator - 32 to 256 plain steps at a time - ran prose 25% slower than plain
decoding while drafting nothing. `SpecPlainStepUsesForward` now routes a PARKED
plain step through the fused decode, and `SpecPlainStepCostsFamilySwitch` keeps
an ordinary no-proposal plain step inside the verify family, where alternating
with verifies is free. Prose came back to within 5% of plain.

## Switching it at run time

`InferenceEngine.UpdateSpeculation(SpeculationOptions)` replaces the policy for
every step from then on without rebuilding the engine: queued like a trim, applied
on the engine thread between steps, the executor drops its armed contexts and
re-arms under the new policy on the next turn (`BatchExecutor.SetSpeculation`).
`InferenceEngineHost.UpdateSpeculation` hands it to the standing engine. It exists
for a settings switch — TensorAgent's applies at once through it — and for an A/B
that must not reload the model between its passes.

## The 2026-09 ggml_cuda regression, and what is left

Speculation on Qwen 3.5-0.8B under `ggml_cuda` measured **5.3x SLOWER** than
plain decoding (108.5 tok/s plain, 20.5 tok/s with the n-gram drafter). Drafting
was not the cost: over that run `draft=1 ms` and `verify=355 ms` against
`plain=4647 ms` for 93 steps, i.e. the steps where the drafter proposed nothing
cost 50 ms/token where the non-speculative engine spends 9.2.

The chain, in the order it has to be read:

1. `LinearSpecTrunk.ForwardPlain` sends an UNPARKED plain step to `SpecForward`
   (the verify graph family) whenever the model declares
   `SpecPlainStepCostsFamilySwitch`. That is right for an isolated plain step
   between verifies and wrong for a long run of them.
2. The governor never parked, so the run never left that family. Its plain
   baseline read **247.4 ms/token** against a true 9.2, because the only plain
   steps it sampled were calibration steps that pay a graph-family transition.
   Against that baseline a 44 ms speculative step is an overwhelming win.
3. Parked steps - the one clean, amortized measurement of plain cost the
   governor ever takes, 32 to 256 of them per park - were explicitly discarded
   as "not samples".

Three changes, none model-specific:

* `SpeculationCostGovernor.Record` now feeds parked steps into the plain
  baseline (skipping the first, which pays the transition).
* `SpeculativeExecution` counts consecutive no-draft steps and, after two,
  takes the model's own decode for the rest of the run - the switch is amortized
  by the run itself.
* The plain branch stopped allocating a vocabulary-sized logits array per token.

A fourth is Qwen 3.5 specific: `ggml_cuda` now keeps the gated-delta-net
recurrent state on the device across a speculative step instead of draining it
to host mirrors and re-uploading it, which is what Metal already did
(`TS_QWEN35_SPEC_DEVICE_STATE=0` restores the drain).

Measured after, same machine, same prompt: **1.13x slower** than plain
(112.4 -> 99.3 tok/s) with the governor parking correctly. Isolating the fourth
change with its kill switch shows it is worth 1.67x on its own (99.3 with it,
59.3 without).

### Measure on a quiet machine, or do not measure

The first pass at the numbers below reported a 246 ms verify step, a 65 ms state
snapshot and 67 ms plain steps, and concluded that the verify path had a
per-call floor no acceptance rate could pay for. **That conclusion was wrong, and
the cause was the measuring environment.** An `nvidia-smi -l 1` left running on
the box for eight hours was taking the driver lock every second; on a 0.8B model
whose decode step is launch-latency bound, that inflated everything and made
otherwise identical runs vary by 3x - plain greedy read anywhere between 105 and
293 tok/s with the same code and the same prompt.

With that process killed, identical runs agree to a few percent. Always check
`ps -eo pcpu,pid,comm --sort=-pcpu` and `nvidia-smi --query-compute-apps` before
believing a speculative measurement.

### What a speculative step actually costs

Qwen 3.5-0.8B Q8_0, ggml_cuda, quiet box, drafting forced on
(`TS_SPEC_ADAPTIVE=0`): 57 drafted, 10 accepted, 19 verifies, 76 plain steps.

| phase | per call |
| --- | ---: |
| verify step | 8.7 ms |
| recurrent-state snapshot | 3.0 ms |
| plain step on the speculative family | 5.3 ms |
| plain step on the non-speculative route | 3.4 ms |

These are sane numbers, and they say where the loss comes from. A verify plus its
snapshot costs 11.7 ms and emits, at 17% acceptance with a 3-token window,
1.53 tokens: 7.7 ms per token against 3.4 ms for a plain decode. The plain steps
on the speculative route cost 5.3 ms rather than 3.4 ms.

So beating plain decoding is an ACCEPTANCE problem, not a floor problem. At
11.7 ms per verify a break-even against 3.4 ms/token needs about 3.4 accepted
tokens per verify.

And on this scenario the drafter cannot get there. Widening the window does not
help at all - tokens emitted per verify stayed at ~1.5 for windows of 3, 8 and 16,
while drafted tokens rose from 57 to 218 - so it is the FIRST drafted token that
is usually wrong, not the tail. Tightening the match instead (`--spec-pmin`,
which scales the required n-gram context length) kills drafting outright: at
pmin 0.5 the drafter proposed 18 tokens in 105 steps, and at 0.75 and 1.0 it
proposed none. A 2-token context matches somewhere in a 2,777-token corpus almost
always, and almost always in the wrong place; a 6-token context never matches at
all. The model is not echoing its prompt verbatim in this scenario, so there is
nothing for a suffix matcher to find.

With the governor on and the default window, speculation measured 0.87x, 0.93x
and 0.94x of plain across three runs, every stream identical. That is the honest
current state: the regression is gone, the remaining 6-13% is the speculative
route's residual per-step overhead, and closing it further needs a workload where
a drafter can actually earn its verify.

### A correctness bug at a nine-row verify (Qwen 3.5, ggml)

Speculation's whole contract is that the emitted stream is what plain greedy
would have produced. On Qwen 3.5-0.8B / ggml_cuda that contract **breaks once the
verify batch reaches nine rows**, which is a draft window of 8 or more.
Reproducible, with `benchmarks/AgentTurnBench --scenarios spec --spec-file 2000`
and `TS_SPEC_ADAPTIVE=0` to force drafting:

| draft window | verify rows | result |
| ---: | ---: | --- |
| 6 | 7 | identical to plain greedy |
| 7 | 8 | identical to plain greedy |
| 8 | 9 | **diverges at token 74** |
| 9 | 10 | **diverges at token 74** |
| 10 | 11 | **diverges at token 74** |

The culprit is the per-row recurrent-state snapshot path. `TS_Q35_VERIFY_SNAPSHOTS=0`
at draft window 8 produces an identical stream; leaving it on diverges. Nine rows
is also where ggml's 2..8-row matvec kernels give way to the large-batch path,
which is the same boundary Gemma 4 already avoids for speed
(`Gemma4Model.SpecPreferredDraftWindow => 7`).

**The default is not affected.** `Qwen35Model.SpecPreferredDraftWindow` is 3 on a
recurrent trunk, and a preferred window narrows the DEFAULT only - it never
overrides a number the operator typed. So this is reachable by passing
`--spec-draft 8` or wider, and not otherwise.

**Workaround until it is fixed:** `--spec-draft 7` or lower, or
`TS_Q35_VERIFY_SNAPSHOTS=0`.

Two C#-side guards were tried and do NOT work, which is worth recording so nobody
repeats them: requesting one snapshot instead of N moved the divergence from
token 74 to token 33, and additionally turning off the deferred state download
moved it to token 18. The native reads `TS_Q35_VERIFY_SNAPSHOTS` itself for
`fv_snapshots_cfg`, so the environment variable disables a combination the
managed side cannot reproduce by toggling its own arguments. The fix belongs in
`ggml_ops_qwen35_verify.cpp`, in how the snapshot slots are captured and
committed for a batch wider than eight rows.

### Device-resident recurrent state: fast and currently wrong

`TS_QWEN35_VERIFY_RESIDENT=1` keeps the gated-delta-net conv and delta state on
the device instead of moving ~60 MB per call. It is a large win on paper and it
breaks the output, which is why it stays opt-in:

* Resident on every call: the stream diverged from plain greedy at token 53. A
  resident call updates the state IN PLACE, so
  `SpecSnapshotRecurrentState`'s shortcut - the live slices ARE the snapshot,
  because a verify only reads them - stops being true, and a rejected draft rolls
  back to nothing.
* Resident on single-token calls only, leaving verifies with separate in/out
  buffers: diverged at token 2, and gave up the plain-step win as well.

Making it correct needs a real snapshot in resident mode. The state is already on
the device, so a device-to-device copy is about 0.2 ms for 60 MB on an A40 - the
expensive part was always the host round trip, not the copy.
`BackendType.Cuda` already has that path (`MtpSnapshotRecurrentStateCudaDevice`);
ggml_cuda does not.

## Measuring it

On the phone, `TensorAgent/scripts/bench-spec-device.sh` deploys the app, launches
it with `TENSORAGENT_SPEC_BENCH=1` and pulls back `specbench.log`: the same four
turns (a one-word answer, prose, a file quoted from the prompt, the same text quoted
from the model's own answer) under plain and speculative decoding, twice each,
switching the engine's policy in place, with prefill and decode rates per turn
(`TensorAgent.Core/Hosting/SpeculationBench.cs`). The Mac host benchmark runs the
same turns as `TensorAgentTtftBench --scenarios spec`, with `--no-spec` for the
other half of the A/B.

`benchmarks/AgentTurnBench` drives the engine — the path the server and
TensorAgent use — with the conversation shapes an agent produces and reports,
per request, the tokens per prefill step, TTFT, prefill and decode rates and the
speculative counters; it compares every speculative stream against plain greedy
token for token. `TS_GMTP_PROFILE=1` prints the Gemma 4 verify's phase timing
(embed+PLE / kernel / norm+head+copy), `TS_SPEC_DRAFT=N` sets the window.

Final Mac numbers on the TensorAgent host path (M5 Pro, ggml_metal, the phone's
settings, a 5k-token system prompt, greedy; `benchmarks/TensorAgentTtftBench
--scenarios spec` against `--no-spec`, three speculative runs each, decode tok/s):

| model | turn | plain | speculative |
| --- | --- | ---: | ---: |
| E2B Q8_0, n-gram | one word (32 tok) | 78 | 72-74 |
| | prose (159 tok) | 77 | 73-74 |
| | quote the prompt (218 tok) | 76 | 145-190 |
| | quote own answer (218 tok) | 78 | 161-179 |
| E4B IQ4_XS + draft head | one word | 65 | 42-44 |
| | prose | 65 | 53-57 |
| | quote the prompt | 65 | 82-100 |
| | quote own answer | 65 | 86-89 |
| Qwen 3.5-9B IQ4_XS, n-gram | one word (1 tok) | - | - |
| | prose | 49 | 46-47 |
| | quote the prompt | 49 | 78-86 |
| | quote own answer | 49 | 89-90 |

Prefill (tokens per second of fresh prompt) is unchanged within noise in every
row: speculation touches decode only.

On the phone (iPhone 17 Pro Max, `TensorAgent/scripts/bench-spec-device.sh`,
Release build, the app's own chat path, 160-token budget, two passes per mode;
mean decode tok/s plain -> speculative, and the first-token cost of the setting):

| model | one word | prose | quote the prompt | quote own answer | first token, spec minus plain |
| --- | ---: | ---: | ---: | ---: | ---: |
| E4B IQ4_XS + draft head, run 1 | 4.8 -> 4.5 | 6.5 -> 8.2 | 7.1 -> 13.3 | 14.1 -> 14.2 | +0.1 to +0.6 s |
| E4B IQ4_XS + draft head, run 2 | 8.4 -> 4.3 | 7.0 -> 6.6 | 12.3 -> 15.5 | 12.6 -> 14.8 | +0.1 to +0.2 s |
| Qwen 3.5-9B IQ4_XS, n-gram, run 1 | - | 7.2 -> 6.6 | 8.2 -> 10.4 | 6.3 -> 10.0 | +0.0 to +0.5 s |
| Qwen 3.5-9B IQ4_XS, n-gram, run 2 | - | 6.5 -> 5.0 | 5.1 -> 9.7 | 6.3 -> 9.7 | +0.1 to +0.2 s |

Read these as shapes, not decimals: the phone runs both models under memory
pressure (a few hundred MB free, the compressor busy), plain decode itself
swings between passes (one plain quote turn stalled at 0.9 tok/s for four
minutes and was dropped from the mean above), and the app samples at
temperature 0.7 so passes answer differently (the "one word" prompt sometimes
gets a 32-token thinking answer). What holds across all four runs: quoting and
echoing decode 1.2-1.9x faster with the setting on; free prose runs 0.8-1.0x
of plain (the governor's probe is dearer on the phone, where a verify is more
expensive relative to a plain step); and every turn pays 0.1-0.6 s more to its
first token with the setting on, which is the per-turn arming - a fresh
holder per chat means fresh verify graphs - and is the part that would repay
work next. E2B was not installed on the phone at the time, so its on-device
row is missing; on the Mac host path it is the strongest of the three. The one-word turn is the first turn of the
process and pays the 8-step probe once; prose is the governor's floor (a probe,
then parked re-probes); quoting is where drafting pays. Qwen 3.5 (n-gram only in the catalog: no draft
head ships for it) speculates on the same holder path since the session fix above.

## Concurrent requests

A multi-sequence step never speculates: the planner rejects both speculative
paths for it, and the per-sequence fused path speculates only when the request
is alone in the engine (a solo step with no other sequence running: while a
newcomer prefills, the lone decoder gets steps of its own between the
newcomer's chunks, and speculating on those delayed the newcomer's first token
by 600 ms on E4B), so the batched kernels (Gemma's token-batched fused decode, Qwen's arena
decode, the paged batched path) serve every running sequence in one graph and
a neighbour's latency stays bounded. Letting the lone decoder keep speculating
beside a newcomer's prefill chunk was tried and measured: the running stream
gained, but the newcomer's first token slipped from 776 to 1,429 ms on E4B
(every chunk step also carried a verify) and on Qwen each such step switched
holders and drained the verify family's state, so the gate stays strictly solo.

A request keeps its holder across a mixed phase, and since 2026-09-09 it keeps
its speculative context too. A solo request starts in the primary linear cache
with a linear speculative context; when a neighbour arrives it is adopted into
a per-request holder, and the context now moves with it (its linear trunk
forwards on whatever cache is bound). Once the request is solo again the
executor resumes that context over the tokens the interlude took plainly
(`SpeculativeExecution.CatchUp` commits them to the drafter with no hidden
rows, which n-gram and a head that resumes after a gap accept; the governor
counts the steps against its park or hold). Before that the context was
dropped and re-armed from scratch on every 1 -> N -> 1 transition - the whole
context re-indexed for n-gram, and a fresh probe of eight verifies - which
staggered short requests hit on every arrival and every completion.

Measured with `benchmarks/AgentTurnBench --scenarios conc --conc 1,2,4
--spec-engine ngram|auto [--conc-stagger 600]` (M5 Pro, Metal, 32-token
answers, decode aggregate tok/s after the last first token; both the engine
default path and the app's holder path give the same picture):

| model | arrival | plain 1 / 2 / 4 | speculative 1 / 2 / 4 | first token, plain vs spec (2 / 4) |
| --- | --- | ---: | ---: | ---: |
| E2B Q8_0, n-gram | simultaneous | 77 / 80 / 88 | 80 / 81 / 87 | 639 vs 634 / 1556 vs 1528 |
| | staggered 600 ms | - / 60 / 50 | - / 55 / 50 | 478 vs 472 / 558 vs 552 |
| E4B IQ4_XS + draft head | simultaneous | 68 / 68 / 73 | 69 / 68 / 73 | 1201 vs 1203 / 2662 vs 2678 |
| | staggered 600 ms | - / 43 / 40 | - / 44 / 40 | 776 vs 764 / 1377 vs 1358 |
| Qwen 3.5-9B IQ4_XS, n-gram | simultaneous | 51 / 59 / 83 | 46 / 58 / 79 | 2228 vs 2182 / 4638 vs 4581 |
| | staggered 600 ms | - / 59 / 34 | - / 58 / 33 | 2182 vs 2150 / 3271 vs 3338 |

Every cell is at parity with the setting off, within run-to-run noise. Two
things had to change to get there, both found with this benchmark. The first
run of the staggered rows showed E2B 5-11% down and, once the engine log was
read, an E4B newcomer's first token at 1,397 ms against 797: the first request
had armed on the linear path at its prefill and run its whole prompt through
the speculative context (33-36% slower than the fused prefill), and the
newcomer waited on it. That is the late-arming change above. The second was the
context churn on every 1 -> N -> 1 transition, now the resume described above.
A third idea - letting the lone decoder keep speculating beside a newcomer's
prefill chunk - was measured and rejected (the newcomer's first token slipped
by 600 ms on E4B), as was speculating on a solo step while another request is
still prefilling; the gate is "alone in the engine".

## Where n-gram pays

`--spec-type ngram` needs no trained weights, so it works on every checkpoint,
including those that ship no speculator at all. It drafts by finding where the
last few tokens occurred earlier in the context and proposing what followed, so
it is strong exactly where the answer quotes its input: summarizing, editing,
translating or answering about a document, repetitive structured output, code
with repeated identifiers, agentic tool loops. On free-form prose it finds
nothing, every step degrades to a plain decode, and the cost governor keeps that
cheap.

Lookup is O(1) per step (an incrementally maintained hash index per n-gram
order), and every hit is verified token by token, so a hash collision can only
cost a rejected draft.
