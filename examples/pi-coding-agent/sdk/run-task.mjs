// Run one coding task with the Pi coding agent as a library, against a local TensorSharp server.
//
//   node run-task.mjs <project-dir> [task]
//
// Environment:
//   PI_CODING_AGENT_DIR  the agent directory holding models.json and settings.json (see ../scripts/pi.sh)
//   TENSORSHARP_URL      the server's OpenAI-compatible base URL (default http://127.0.0.1:5000/v1)
import path from "node:path";
import { createAgentSession, ModelRuntime, SessionManager } from "@earendil-works/pi-coding-agent";

const projectDir = path.resolve(process.argv[2] ?? ".");
const task = process.argv[3] ??
  "Run the test suite with `npm test`. Fix the bugs in src/stats.js so that every test passes. " +
  "Do not change the tests. Run the tests again to confirm, then summarize the fixes in two sentences.";
const agentDir = process.env.PI_CODING_AGENT_DIR;
if (!agentDir) {
  console.error("Set PI_CODING_AGENT_DIR to the agent directory with models.json (scripts/pi.sh prepares one).");
  process.exit(2);
}

// The model runtime reads <agentDir>/models.json, where the "tensorsharp" provider points at the server.
const modelRuntime = await ModelRuntime.create({
  modelsPath: path.join(agentDir, "models.json"),
  authPath: path.join(agentDir, "auth.json"),
});
const model = modelRuntime.getModel("tensorsharp", "Qwen3.8-27B-UD-IQ3_XXS");
if (!model) {
  console.error(`No tensorsharp/Qwen3.8-27B-UD-IQ3_XXS in ${agentDir}/models.json.`);
  process.exit(2);
}
if (process.env.TENSORSHARP_URL) model.baseUrl = process.env.TENSORSHARP_URL;

const { session } = await createAgentSession({
  cwd: projectDir,                          // the built-in read/bash/edit/write tools work here
  agentDir,
  modelRuntime,
  model,
  thinkingLevel: "medium",                  // sent as chat_template_kwargs.enable_thinking = true
  sessionManager: SessionManager.inMemory(), // no session files for a one-off run
});

const usage = { input: 0, output: 0, cacheRead: 0, requests: 0 };
const started = Date.now();
session.subscribe((event) => {
  switch (event.type) {
    case "message_update":
      if (event.assistantMessageEvent.type === "text_delta") process.stdout.write(event.assistantMessageEvent.delta);
      break;
    case "tool_execution_start":
      process.stdout.write(`\n[tool] ${event.toolName} ${JSON.stringify(event.args ?? {}).slice(0, 160)}\n`);
      break;
    case "tool_execution_end":
      process.stdout.write(`[tool] ${event.toolName} ${event.isError ? "failed" : "done"}\n`);
      break;
    case "message_end":
      if (event.message.role === "assistant" && event.message.usage) {
        usage.input += event.message.usage.input ?? 0;
        usage.output += event.message.usage.output ?? 0;
        usage.cacheRead += event.message.usage.cacheRead ?? 0;
        usage.requests += 1;
      }
      break;
  }
});

try {
  await session.prompt(task);
  const seconds = (Date.now() - started) / 1000;
  console.log(`\n\n--- final answer ---\n${session.getLastAssistantText()}`);
  console.log(`\n--- ${usage.requests} model requests in ${seconds.toFixed(1)} s: ` +
    `${usage.input} prompt tokens (${usage.cacheRead} served from the server's prefix cache), ${usage.output} generated ---`);
} finally {
  session.dispose();
}
