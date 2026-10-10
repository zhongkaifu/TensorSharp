This replay checks how the Nemotron-H output parser splits logged replies.
It needs no model and no GPU. It reads the `chat.start` / `chat.complete`
lines of TensorSharp.Server or TensorAgent logs. It replays each logged
Nemotron-H `assistantOutput` through the parser that `ChatProtocolRegistry`
creates. Each reply is fed in pieces of 1, 3 and 7 characters and as the whole
text, once held (as an API stream or the CLI consumes it) and once retracting
(as the Web UI does).

```bash
dotnet run --project eng/validation/NemotronThinkOffReplay -- \
  --out artifacts/nemotron-thinkoff-replay \
  --opened server.log:440-760 \
  artifacts/e2e-1009/nemotron3.log artifacts/nemotron-tools-ab/server.log
```

The log does not show the prompt's tail; the request flag stands in for it.
`--opened LOG:FROM-TO` marks the requests (by their `chat.start` line) whose
system prompt carried `{'reasoning': True}`. That marker opens the block even
though the log says `thinking=False`. A bare file name matches every log with
that name; use a path when two logs share one.

The tool checks the following for every reply:
- Tool calls match today's parse and surface at the same character.
- The held and retracting parses end with the same split.
- Every streamed split decides the reply as the whole text does (answer,
  reasoning and calls), as the transcript and the streams must agree.
- Every retraction is the exact end of the answer shown so far.
- Replies that need no change are parsed as before, and the retracting stream
  shows their first answer character at the same position as before.
- For a stray close inside the window, `</think>` is never shown, and the
  reasoning before it becomes thinking.

It also reports how many logged stray closes the window covers and how much
later a held stream shows an ordinary answer. A nonzero exit code means at
least one check failed. Write reports under ignored `artifacts/` or
`docs/validation/`.
