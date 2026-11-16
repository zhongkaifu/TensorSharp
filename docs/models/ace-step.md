# ACE-Step 1.5 (music generation)

[English](ace-step.md) | [中文](ace-step_zh-cn.md)

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (Apache-2.0) writes songs, sung or instrumental, from a
style caption, lyrics and optional tempo, key and time signature, at 48 kHz stereo. TensorSharp runs the turbo
checkpoint the way ComfyUI's template does, on `--backend cuda` (resident BF16 weights; `--backend cpu` works, slowly):

1. a Qwen3 planner LM writes five audio codes per second from the caption, the lyrics and the metadata (its chain of
   thought holds the metadata as YAML), with classifier-free guidance 2 against an empty chain of thought,
   temperature 0.85 and top-p 0.9;
2. Qwen3-0.6B encodes the caption and metadata; the lyrics enter as its token embeddings, the timbre as the
   reference "silence" latent, and the codes as hints decoded through the FSQ quantizer and the detokenizer;
3. a 2.4B DiT (24 blocks, AdaLN, sliding-window self-attention on alternate blocks, cross-attention over the
   condition) denoises the 25 Hz, 64-channel latent in 8 Euler steps on the flow schedule with shift 3;
4. the Oobleck VAE decodes it (on the CPU, in overlapping tiles), and the result is scaled down when its standard
   deviation is above 0.2, as `VAEDecodeAudio` does.

## Make the GGUF

From ComfyUI's all-in-one checkpoint (`ace_step_1.5_turbo_aio.safetensors`, Comfy-Org/ace_step_1.5_ComfyUI_files)
and a ComfyUI checkout, which supplies the Qwen tokenizer files and the silence latent:

```bash
python eng/audio-models/acestep_to_gguf.py --input ace_step_1.5_turbo_aio.safetensors --comfyui path/to/ComfyUI \
    --out ace-step-1.5-turbo.gguf
```

The parts keep their release precision (BF16, the text encoder F16); the VAE decoder is weight-norm folded to F32.
The VAE encoder and the audio tokenizer's pooler, used only for reference audio, are left out (9.4 GiB in all).

## Run it

```bash
TensorSharp.Cli --model ace-step-1.5-turbo.gguf --backend cuda --prompt "upbeat lofi hip hop, mellow piano" \
    --lyrics "[Instrumental]" --duration 30 --bpm 90 --key "C major" --time-signature 4 --diffusion-seed 7 \
    --output song.wav
```

`--lyrics` takes text or a file, with `[verse]` / `[chorus]` section tags; `--language` names the lyrics' language
(`en`, `zh`, ...). Leaving out `--bpm`, `--key` or `--time-signature` leaves them to the planner. `--diffusion-steps`
overrides the 8 steps. The server route is

```bash
curl http://localhost:5000/v1/audio/music -H "content-type: application/json" \
    -d '{"prompt": "lofi hip hop, piano", "lyrics": "[Instrumental]", "duration": 30, "bpm": 90, "seed": 7}' -o song.wav
```

## What was checked

Against ComfyUI running the same checkpoint in FP32, with ComfyUI's planner codes given to TensorSharp
(`TS_ACESTEP_CODES`; the planner samples with torch's generator, which is not reproduced) and the same seed:

| Stage | `--backend cpu` | `--backend cuda` |
|---|---|---|
| Caption encoder states | relative error 3e-6 | 7e-4 |
| Lyric embeddings | exact | exact |
| Final latent after 8 steps | relative error 9.5e-4, correlation 1.000000 | 1.6%, correlation 0.99988 |
| Decoded audio | correlation 0.99998 | 0.9977 |

The CUDA differences are BF16 activations against an FP32 reference. The planner's first-step logits, conditional
and unconditional, correlate with ComfyUI's at 0.9999 with the same top 10 codes. A 10-second song takes about 15 s
on one GPU (planner 5 s, encoders 1 s, diffusion 0.18 s per step, VAE 7 s on the CPU); the VAE is the part to move to
the GPU next.

Not implemented: reference audio (covers, repainting), negative captions, and the 4B planner variant.
