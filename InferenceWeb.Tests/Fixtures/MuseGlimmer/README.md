# Muse-Glimmer fixtures

## Chat template

`chat_template.jinja` is the unmodified `tokenizer.chat_template` metadata from
`Muse-Glimmer-30B-UD-Q4_K_XL.gguf`, inspected on 2026-10-11. It ignores
`enable_thinking` and ends every generation prompt with a bare assistant header.
The fixture exercises the actual template's reasoning, recipient, and tool
branches without requiring model weights or a native backend.

## llama.cpp goldens (`LlamaCpp/`)

`MuseGlimmerParityTests` checks Muse-Glimmer against these. They were made on
2026-11-20 by `eng/validation/muse-glimmer-llamacpp-goldens.py` with llama.cpp
**d3c3e242c** (a clean Release Metal build; it includes the upstream Muse Glimmer
vision fix #32433, and its text graph is 1a24df163's after a refactor) from
`Muse-Glimmer-30B-UD-IQ2_XXS.gguf` (sha256 `3e9bd0e6...`, unsloth/Muse-Glimmer-30B-GGUF
at `faa5b025`). Each file records the llama.cpp commit, the server arguments and the
model's sha256.

| File | Contents |
|---|---|
| `ref_text.json` | five chat prompts (English prose, the Chinese prompt that looped in September 2026, code, arithmetic, verse): the prompt text, llama.cpp's prompt tokens, its 128-token greedy continuation, and each generated token's top log-probabilities (10 at the first position, 4 after) |
| `ref_text_long.json` | a 4,664-token prompt (more than twice the 2048-token window and past legacy's 4352-row ring) and its continuation, as above |
| `ref_text_dflash.json` | the five prompts through llama.cpp's DFlash path (`dflash-kquant.gguf`, `--spec-draft-n-max 15`): every continuation is its plain one |

The prompts are rendered by the template's rules without its date line, so a
regeneration reproduces them. Regenerate (about 90 s on an M5 Pro, peak 16 GB wired):

```bash
python3 eng/validation/muse-glimmer-llamacpp-goldens.py --llama-cpp ~/work/llama.cpp-lfm3 \
    --model Muse-Glimmer-30B-UD-IQ2_XXS.gguf --draft dflash-kquant.gguf
```
