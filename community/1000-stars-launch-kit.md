# TensorSharp: community launch kit

This kit is a reviewable set of launch materials. Posts are **drafts**, not claims of publication. Adapt each to the community rules, disclose project affiliation, and respond to technical questions. Do not mass-post identical text, buy stars, or ask for stars in exchange for rewards.

## Positioning

**TensorSharp: local GGUF inference and agent workflows from .NET.** Run supported text, multimodal, image and video models on Windows, macOS and Linux through a CLI, browser chat, compatible APIs or a C# integration. Backend and model support vary; see [supported models](../docs/supported_models.md) and [project status](../docs/PROJECT_STATUS.md).

### Landing-page checklist

- Lead with one sentence on the developer problem, followed by one clickable demo and the release download.
- Make a reproducible quick start available before the long model catalog.
- Link measured benchmarks with hardware, quantization, version, prompt length, output length, TPS and limitations.
- Show a compact “when to use TensorSharp / when to use another engine” comparison.
- Track visits to documentation and release downloads using privacy-respecting aggregate analytics where available.

## Draft 1: r/dotnet

**Title:** I’m building TensorSharp, a native .NET inference engine and agent runtime for local GGUF models

**Body:** I’m the developer of [TensorSharp](https://github.com/zhongkaifu/TensorSharp). It runs supported local GGUF models through C#/.NET 10, with CLI, browser chat and OpenAI/Ollama-compatible endpoints. The project also includes agent skills, local knowledge collections, and model-specific image/video features. CUDA, Metal, Vulkan and CPU are available, with support varying by model and platform.

I’d especially value feedback from .NET developers on the integration experience, packaging and what would make it useful in production. The [getting-started guide](https://github.com/zhongkaifu/TensorSharp/blob/main/docs/getting_started.md) and [limitations](https://github.com/zhongkaifu/TensorSharp/blob/main/docs/PROJECT_STATUS.md) are public. What would you want to see in a native C# inference stack?

## Draft 2: r/LocalLLaMA

**Title:** How we approach SSD/RAM/VRAM scheduling for local MoE inference in TensorSharp

**Body:** I’m working on [TensorSharp](https://github.com/zhongkaifu/TensorSharp), a .NET GGUF inference engine. One focus is getting large MoE models to run on constrained machines through coordinated placement and offloading. I’d like to share the memory-placement design, where it helps, and the trade-off between model capacity and token throughput.

Before posting benchmark numbers, I’ll attach the exact GGUF, hardware, quantization, engine commit, context, cold/warm conditions, TTFT and decode TPS. Offloading can make a model fit without making it fast; I want the discussion to focus on reproducible results and optimization ideas. What would be the most useful comparison configuration?

**Publication gate:** Add a fresh reproducible measurement and charts before posting. Do not imply full-precision or full-speed inference from a low-bit offload demonstration.

## Draft 3: r/aiagents

**Title:** Building local browser automation with a GGUF model and Playwright

**Body:** I’m building [TensorSharp](https://github.com/zhongkaifu/TensorSharp), a local inference engine and agent runtime. I’ve been exploring how local model inference, tool calls and Playwright browser automation work together on real tasks. The goal is to keep model inference and agent state on-device while browser interactions go to the sites being automated.

I’d like to share a short, uncut task demo alongside the model, hardware, prompts, tool traces, failure cases and which steps required human intervention. I’m interested in how others measure reliability, permission boundaries and recovery from bad browser actions.

**Publication gate:** Attach the demo and accurate end-to-end timing before posting. Never suggest the system completes payments autonomously if final approval was manual.

## Outreach and experiment plan

1. Establish a daily baseline: stars, forks, visitors, unique cloners, release downloads, issues and community referral sources. Some GitHub traffic metrics require maintainer access and only cover a limited historical window.
2. Publish the improved landing page and one technical post. Do not cross-post duplicates in rapid succession.
3. Answer substantive questions and incorporate feedback in documentation.
4. Publish a second distinct demo after the first discussion has matured.
5. Review 48-hour traffic and star deltas, and double down on sources producing engaged users rather than raw clicks.

## Suggested reporting table

| Date (UTC) | Stars | New stars | Repo visitors | Release downloads | Source / post | Notes |
|---|---:|---:|---:|---:|---|---|

**Success metric:** 1,000 genuine stars. **Quality guardrails:** reproducible claims, no spam, no misleading benchmarks, no artificial engagement.
