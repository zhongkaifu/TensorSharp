---
name: unified-agentic-probe
description: A local validation skill that computes a small checksum and writes a real execution receipt. Use it only for the unified-memory agentic acceptance test.
---

Read references/input.json, then invoke skills_run with skill="unified-agentic-probe", path="scripts/run_probe.py", and args containing the nonce from the user's request. Do not copy or rewrite the script. It calculates the sum of the input values, writes skill-receipt.json in the current workspace, and prints its JSON. Report its checksum and receipt link. It uses only the standard library, performs no networking, and installs nothing.

For read-only delegated tasks, references/alpha.txt and references/beta.txt contain the authoritative facts. Read the requested file and return its exact FACT line.
