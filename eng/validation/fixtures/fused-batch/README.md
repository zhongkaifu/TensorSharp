# Production logger regression excerpt

`actual-server-v3-excerpt.jsonl` contains the first six mapping/acceptance records
from the 2026-10-11 `fused-server-v3` qualification. It preserves two startup
bindings without HTTP owners, two actual HTTP bindings, and two completed native
joint graphs. The production formatter represents native UInt64 execution IDs
as strings and flattens nested scope fields in their original order.

Source evidence SHA-256:
`2ae12c533220e9f67fb9d208c14736d3ba30fa1b9ed05b450d4308152db854d1`.

This small required test fixture validates parser behavior. It is not a new
model run, complete quality report, benchmark, or substitute for the full
ignored original evidence.
