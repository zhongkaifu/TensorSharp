# Muse-Glimmer parity goldens

The llama.cpp goldens `InferenceWeb.Tests/MuseGlimmerParityTests.cs` checks against are
committed test fixtures now: `InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/`
(see the README there for their provenance). Regenerate them with
`eng/validation/muse-glimmer-llamacpp-goldens.py`, which starts its own `llama-server`.

The parity tests feed the RECORDED prompt token ids into the model and teacher-force
llama.cpp's continuation, so every generated position is compared apart from the
tokenizer, which has its own test. They need `TS_TEST_MODEL_DIR` (a directory holding
`Muse-Glimmer-30B-UD-IQ2_XXS.gguf`, and `dflash-kquant.gguf` for the DFlash test) and
skip without it; `TS_MUSE_GLIMMER_BACKEND` picks the backend (default: the GGML backend
the test process pins).

`bench_mg_README.md` describes the 2026-08-13 CUDA benchmark harness (the scripts are
not in the repository).
