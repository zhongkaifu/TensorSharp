# Explicit Prompt Cache Markers

TensorSharp caches prompt prefixes automatically. A client can instead dictate where
cacheable segments end with `cache_control` markers: a request that carries at
least one marker switches from *automatic* caching (everything is cacheable) to
*explicit* caching (nothing past the request's last marker is stored or reused).
This page describes what the code does.

## 1. Accepted markers

`"cache_control": {"type": "ephemeral"}` or `"prompt_cache_breakpoint": true`
(`TensorSharp.Chat/RequestParsers/CacheControlParser.cs`).

* A marker may sit on a message or on a text content part in
  `/v1/chat/completions` messages and `/v1/responses` input items.
* A marker may sit on a tool declaration in OpenAI chat, Responses and Ollama
  requests. A marker on any tool marks the whole rendered tool block.
* Ollama chat messages are not parsed for markers.
* An unrecognised `type` is kept verbatim and acts as a plain breakpoint.

A request opts in only by carrying a marker. No request header switches caching
modes (`X-Prompt-Cache-Control` and `X-DashScope-CacheControl` are not read).

## 2. Rendering and marker tracking

Markers must not change the rendered prompt, yet the engine needs their exact
token offsets. `KVCachePromptRenderer` therefore:

1. inserts an invisible breakpoint sentinel (``, `KVCachePromptRenderer.MakeBreakpoint`)
   where each marked content part or tool block ends;
2. tokenizes the whole prompt;
3. records the token index of each sentinel and strips the sentinels, so they never
   reach the model.

The rendered token sequence is identical to an unmarked render. The offsets reach
the engine as `SequenceState.CacheBreakpoints`; when a prompt is truncated they move
with it.

## 3. Capture and reuse

The radix prefix cache enforces the boundaries:

* A marked request neither stores nor adopts cached state past its last breakpoint
  (`SequenceState.CacheBreakpointLimit`). A policy with no usable breakpoint (an
  empty list, or a single breakpoint at 0) disables prefix reuse for that request.
* **Page families** (Mistral 3, Hunyuan Dense, Muse-Glimmer, Nemotron-H)
  store whole KV blocks only, so a marker at token `M` makes `floor(M / block size)`
  blocks reusable; the tail of the segment is recomputed.
* **Checkpoint families** (the models that restore from state checkpoints) end a
  prefill chunk at each breakpoint (`ContinuousBatchScheduler.AlignSharedPrefixBoundary`)
  and capture a checkpoint there, so the marked prefix is restorable exactly.
* A marked request is served only through those two paths. The model's live cache
  is not continued for it, persisted shared-prefix checkpoints are not restored for
  it, and its finished state is not retained, so no path can reuse past a client's
  boundary.

## 4. Retention priority

A checkpoint captured at an explicit breakpoint is flagged `EndsAtBreakpoint` and
placed in the `Breakpoint` eviction tier, which is evicted after retired and
ordinary entries. This protects long-lived agent conversations from background
traffic. Page-family captures have no such priority.

## 5. Usage reporting

Clients measure cache effectiveness from `cached_tokens`, so it is always reported,
including when it is 0:

* chat completions report `usage.prompt_tokens_details.cached_tokens` on the
  non-streaming response and on the final streaming chunk
  (`OpenAIResponseFactory`);
* Responses report `usage.input_tokens_details.cached_tokens`
  (`OpenAIResponsesFactory`).

```json
"usage": {
  "prompt_tokens": 1000,
  "completion_tokens": 50,
  "total_tokens": 1050,
  "prompt_tokens_details": { "cached_tokens": 768 }
}
```

The value is `InferenceCompletion.PrefixCacheReusedTokens`, carried through
`ChatGenerationPipeline` as `kvCacheReusedTokens`.

## 6. Tests

* `KVCachePromptRendererTests`: marker tracking and stripping.
* `RadixPagedEngineTests.ExplicitPolicyCapsReuseAndEmptyPolicyDisablesIt`: the cap
  and the cache-none policy on page families.
* `RadixHolderEngineTests`: breakpoint-aligned prefill and capture on checkpoint
  families.
* `PrefixTreeMatchTests.BreakpointLimit_CapsTheMatch`,
  `PrefixTreeInsertSplitTests.Insert_BreakpointAndPromptEndFlags_AreRecorded` and
  `PrefixTreeEvictionTests`: the tree's limit, flag and `Breakpoint` tier.

None are organised as the specification's V1–V8 conformance vectors.
