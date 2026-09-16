# Trained Qwen3.8 HTTP replay — r4

**Failed and not release-qualified.** The actual trained Qwen3.8 UD-Q2_K_XL checkpoint ran on physical A40 GPUs 0/1/5 with r4 native SHA256 `7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`, the shared Q8 head attached but speculation off, and BF16 vision projector. Layer placement is not weight tensor parallelism. Timing is descriptive on a shared host.

Recorded scenario results:

| Suite | Result |
|---|---|
| API contract | 32/32 pass |
| Original quality | 37/40 pass; three agentic final answers contain Markdown JSON fences |
| Structured tool results | 10/10 pass |
| Tool policies | 26/29 pass; three of five requests for two calls return only one |
| Static images | 20/20 pass, including follow-up and 8K history |
| Repeated decode | 29/30 pass; one short concurrent answer uses its 512-token budget before mentioning collisions |
| Repeated 8K decode subset | 15/15 scenario checks pass; cross-native full-output comparison remains outstanding |
| Original retrieval | 8K and32K pass;67,154-token over-limit input remains failed |
| Ordered video history | Codes17/86 pass in both modes; timestamp follow-up fails in both (1 instead of2) |
| Replacement65,065-token launcher | Fails before HTTP because helper model argument differs from the pinned body's model string; not a model run |

All model/head/media file hashes and metadata are unchanged before/after; see [owner](owner.json). The exact mapped native and managed application identity was checked at readiness and between suites. The final whole-application directory check **failed** because the server added runtime logs and uploaded media inside the deployment. It reports no missing or changed preexisting files, but this does not turn the original strict audit into a pass. See [profile record](lifecycle/profile.json). Subsequent runs must redirect logs and uploads outside their pinned applications and repeat the checks.

The required/named tool cases happen to pass their prompts. Source inspection found that Qwen validates declaration membership but lacks generation constraints for those policies; the campaign does not prove enforcement. Malformed-policy input validation is addressed separately in [tool-policy-r1](../tool-policy-r1/README.md).

The video timestamp correction and corrected counted-long launcher have separate run records. Original failure evidence and oracles above remain unchanged. Static-image passes do not establish general video support; successful retrieval alone does not establish independent full-checkpoint numerical QSA parity.
