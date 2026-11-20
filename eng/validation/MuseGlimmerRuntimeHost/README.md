# Muse-Glimmer runtime host

Runs `TensorSharp.Server.Host` (or, with `cli` first, `TensorSharp.Cli`) with its own
arguments and code, except that a Muse-Glimmer GGUF loads as the unified-runtime family
(`MuseGlimmerRuntimeModel`) instead of the legacy `MuseGlimmerModel`. Until stage S8 of
`docs/design/muse-glimmer-unified.md` points the architecture descriptor at the runtime
family, this is how its serving gates (stage S3: multi-turn prefix reuse, concurrent
requests, checkpoint files) run through the real server, its Web UI, OpenAI and Ollama
routes, and the CLI. It sets the internal seam
`MuseGlimmerArchitecture.DescriptorBuildsTheRuntimeFamily` before the host's entry point
runs; no option, switch or environment variable of the shipped hosts reaches it. Stage S8
deletes the seam and this tool.

Build it into the server's output directory, which holds the Web UI, the skills and the
native libraries the hosts load from their base directory:

```bash
TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/validation/MuseGlimmerRuntimeHost -c Release -o TensorSharp.Server.Host/bin
H=TensorSharp.Server.Host/bin/MuseGlimmerRuntimeHost.dll
M=Muse-Glimmer-30B-UD-IQ2_XXS.gguf

# The server (every TensorSharp.Server.Host option), the agent loop off for plain generations:
dotnet $H --model $M --backend ggml_metal --port 5120 --no-multi-agent --no-skills

# The CLI (every TensorSharp.Cli option):
dotnet $H cli --model $M --backend ggml_metal --no-skills --multi-turn-jsonl turns.jsonl
```

The first stderr line says which family loads. The legacy arm of an A/B is the shipped host
itself (`dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll ...`), from the same
build. `eng/validation/multiturn-reuse.py`, `parallel-multiturn-webui.py` and
`parallel-reuse-check.py` drive either.
