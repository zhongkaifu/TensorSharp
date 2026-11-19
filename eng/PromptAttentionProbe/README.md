# PromptAttentionProbe

Measures the CUDA executor's prompt attention on one layer of one sequence's prompt chunk: the 16-bit prompt kernel
(`ts_flash_prompt_f16_d256`) against the 8-bit pair (`ts_flash_prompt_band_f16_d256` + `ts_flash_prompt_i8e4_d256`,
with the split combine), each with the split rule the executor gives it, on synthetic operands (24 or 16 query heads over
4 KV heads of 256). It needs an sm_120a / sm_121a build of `tensorsharp_flash_prefill` (an nvcc build on an RTX Blackwell
GPU or GB10) and refuses to run elsewhere.

```
dotnet run -c Release --project eng/PromptAttentionProbe -- kernel --repeats 5 --output artifacts/prompt-attention/kernel.json
```

- `kernel` (the only mode so far): the three shapes of issue #376 (968 rows over 47,981 keys, 2,048 over 67,584, 968
  over 25,104), then a sweep of rows {17, ..., 1024} x keys {8K, 32K, 64K} x GQA groups {6, 4} (`--sweep 0` skips it).
  Each time is the median of `--repeats` launches timed with CUDA events after a warm-up launch; the 8-bit total
  includes the band. It prints a Markdown table and, after the sweep, the smallest row count from which the 8-bit pair
  wins at every swept key count (the rule `CudaFlashPrefillKernels.Attn8MinRows` is set by, per architecture, by hand).
- Timings are printed and written (`--output`, JSON), never asserted. Write outputs under the ignored `artifacts/`.
