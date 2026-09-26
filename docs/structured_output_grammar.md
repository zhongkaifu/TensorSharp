# Grammar-constrained structured output

On `/v1/chat/completions`, TensorSharp enforces `response_format` during
decoding rather than only checking it afterwards (see
[Limits worth knowing](#limits-worth-knowing) for `/v1/responses`). The schema is compiled into a grammar, and at every step the tokens
that would break that grammar are removed from the distribution before sampling.
Invalid JSON is not repaired — it is never generated.

## Why this replaced prompt-and-repair

The previous implementation asked the model for JSON in the system prompt, then
extracted the first balanced object from whatever came back. That can salvage a
markdown fence or a chatty preamble, but it cannot make the model use the right
types, respect an enum, honour `required`, or leave out a property the schema
forbids. Any consumer that depends on the shape — function calling above all —
fails on the difference.

Constrained decoding removes the failure mode instead of compensating for it: a
token that cannot continue a valid document has no probability mass to be
sampled from.

## Measured

Qwen3.5-9B-Q8_0 on an RTX 3080 Laptop, `ggml_cuda`, temperature 1.3 / top-p 0.98,
8 adversarial prompts × 4 repeats = 32 requests per cell. The prompts actively
invite trouble: they ask for preambles, markdown fences, extra fields, and
out-of-enum values. Schemas use closed objects (`additionalProperties: false`),
enums, bounded integers, `pattern`, and nested objects.

| Mode | Constrained | Valid JSON | Schema-conformant | Decode |
|---|---|---:|---:|---:|
| `json_schema` | **on** | **32/32** | **32/32** | 25.8 tok/s |
| `json_schema` | off | 29/32 | 29/32 | 25.0 tok/s |
| `json_object` | **on** | 32/32 | 32/32 | 30.7 tok/s |
| `json_object` | off | 32/32 | 32/32 | 30.6 tok/s |

Reading it: `json_schema` is where the guarantee pays — three responses that the
repair layer let through as non-conformant are now impossible. `json_object` is
the easy case, and the old path already handled it; the value there is that it
now holds by construction rather than by luck. **Throughput is unchanged**, which
is the point of the caching described below.

Masking cost, measured against the real 248,320-token vocabulary:

| Stage | Cost |
|---|---|
| Vocabulary trie build (once per model) | 466 ms, 592,073 nodes |
| Token mask, first visit to a parser state | ~30 ms |
| Token mask, cached state | 0.03 µs |
| Applying a mask to 248,320 logits | 45 µs / token |

At 30 tok/s decode, 45 µs is 0.14% of a step.

## How it works

Three pieces, in `TensorSharp.Runtime/Grammar/`:

1. **Grammar + GBNF parser** (`Grammar.cs`, `GbnfParser.cs`) — llama.cpp's GBNF
   element model, so grammars are portable between the two engines. Rules are
   flattened into one array so a parse position is a single `int`.
2. **Pushdown matcher** (`GrammarMatcher.cs`) — tracks the *set* of viable parse
   stacks, because a grammar is ambiguous in general. UTF-8 is decoded
   incrementally, so a multi-byte character split across byte-fallback tokens
   still matches.
3. **Token masking** (`GrammarTokenVocabulary.cs`, `GrammarConstraint.cs`) — see
   below.

`JsonSchemaGrammarCompiler.cs` compiles JSON Schema to GBNF, with
`RegexToGbnf.cs` translating the supported `pattern` subset, and
`GrammarLibrary.cs` caches the result per (grammar, tokenizer). The same
machinery also constrains DeepSeek V4.1's DSML tool calls on
`/v1/chat/completions` (`DeepSeek41ToolGrammar.cs`; see the
[DeepSeek V4.1 card](models/deepseek41.md#chat-tools-and-json)).

### The masking layer

The obvious implementation — llama.cpp's — tests all V tokens against the grammar
every step. At a 248k vocabulary that is 248k grammar walks per generated token.
TensorSharp instead borrows the structure xgrammar and outlines use:

- **A byte trie over the vocabulary.** Tokens share prefixes, so a first byte is
  tested once for every token beginning with it, and rejecting a prefix prunes
  its whole subtree.
- **A mask cache keyed on the parser state.** JSON generation cycles through a
  handful of states ("expecting a key", "inside a string", "after a comma"), so
  in steady state producing a mask is a dictionary lookup.
- **A transition memo**, so the trie walk does not re-derive the same character
  transition once per node.

The trie mask is verified against the brute-force per-token reference in the test
suite and was checked to agree exactly on all 248,320 tokens across seven parse
prefixes.

## Supported schema keywords

Enforced: `type` (single or union), `enum`, `const`, `properties`, `required`,
`additionalProperties`, `items`, `prefixItems`, `minItems`/`maxItems`, `anyOf`,
`oneOf`, shallow `allOf`, `$ref`/`$defs` including recursive schemas,
`minLength`/`maxLength`, `pattern` (regex subset), the `date` / `time` /
`date-time` / `uuid` formats, and `minimum`/`maximum` on integers.

Refused with a 400, because a context-free grammar cannot express them: `not`,
`if`/`then`/`else`, `dependentSchemas`, `dependentRequired`, `multipleOf`,
`patternProperties`. Refusing beats silently ignoring — a caller is never told a
constraint was applied when it was not.

**Anything not modelled is relaxed, never approximated.** An unsupported regex
construct widens to an unconstrained string rather than compiling to a
nearly-right grammar. A grammar that is too tight is far worse than one that is
too loose: it makes valid output unreachable and the request fails mid-string.

## Limits worth knowing

- **The grammar cannot guarantee the document fits in `max_tokens`.** End-of-
  sequence stays masked until the JSON is complete, so a budget that is too small
  truncates mid-object and the response is invalid after all. Measured: at
  `max_tokens: 300` and temperature 1.3, prompts asking for "a long preamble"
  filled the budget inside a string value. Give structured requests headroom.
- **Property order is fixed** to the schema's declared order. Permuting *n*
  optional keys would need *n!* alternates.
- **`oneOf` is compiled as `anyOf`.** "Exactly one" needs negation.
- Under `--tp N` and on non-GGML backends the constraint applies normally; it
  operates on logits and is independent of the compute backend.
- **Only `/v1/chat/completions` attaches the grammar.** On `/v1/responses`,
  `text.format` gets the same prompt instruction and output validation (HTTP 422
  on a non-streaming request when the result does not conform; a streamed
  response ends with `response.failed`), but decoding is not constrained. There,
  `text.format` cannot be combined with reasoning or with tools (HTTP 400).
- **`response_format` cannot be combined with active `tools`** (HTTP 400).
  Sending `"tool_choice": "none"` drops the tools, so a request that still
  carries its tool catalog can ask for JSON after a tool round. Requested
  skills are allowed and are written into the prompt instead of being offered
  as tools. The prompt instruction and output validation still run alongside
  the grammar, and a non-conforming result returns HTTP 422 on a non-streaming
  or buffered response (`json_schema`, or `json_object` with
  `TS_STRUCTURED_STREAM_BUFFER=1`); an unbuffered streamed `json_object` is
  filtered to the balanced object but not validated.
- **`response_format` with `"think": true`** is accepted only on families whose
  protocol declares where the answer starts after reasoning: Gemma 4, Qwen 3.8
  Flash Next (`qwen4exp`), GPT-OSS, Muse-Glimmer, DeepSeek V4.1, GLM-5.3-Flash
  (`glm5next`) and Nemotron-H. The grammar stays idle while the model reasons
  and arms at that marker (for example `</think>`), skipping whitespace after
  it. Other families return HTTP 400, and so does any `think` request when
  `TS_JSON_GRAMMAR=0`.
- **A schema the compiler cannot turn into a grammar** falls back to the
  first-token nudge, and a warning is logged. The exception is a `think`
  request on the families above, which fails instead, because the delayed
  grammar is required there.

## Controls

| Setting | Effect |
|---|---|
| `TS_JSON_GRAMMAR=0` | Fall back to prompt-and-repair (A/B testing) |
| `TS_JSON_FORCE_OPEN=0` | Disable the first-token nudge used by that fallback |
| `TS_STRUCTURED_STREAM_BUFFER=1` | Buffer a streamed `json_object` response as well. A streamed `json_schema` response is always buffered and schema-normalized before it is sent; `json_object` streams token by token by default |

## Using it

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0",
    "messages": [{"role": "user", "content": "Weather in Lima, as json."}],
    "max_tokens": 600,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "weather",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "location":      {"type": "string"},
            "temperature_c": {"type": "integer", "minimum": -60, "maximum": 60},
            "conditions":    {"type": "string", "enum": ["sunny","cloudy","rain","snow"]}
          },
          "required": ["location", "temperature_c", "conditions"],
          "additionalProperties": false
        }
      }
    }
  }'
```

A raw GBNF grammar can also be used directly from code:

```csharp
var grammar = Grammar.Parse("root ::= \"yes\" | \"no\"");
samplingConfig.Grammar = new GrammarConstraint(grammar, tokenizer);
```
