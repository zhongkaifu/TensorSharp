# CPU audit of retained T8 attention evidence

This tool never registers CUDA, creates a context or calls a device function.
It rehashes all component/timing CAS objects and sidecars, admits the complete
624/1680-kernel campaigns, checks every retained output against the frozen scoped
FP64/analytic oracle, checks dynamic graph evidence, and recomputes all paired
captured-full-chain medians and wins. It independently compares complete timing
bytes to each qualified arm.

For strict frozen-producer replay, it loads the actual R23 probe DLL in an
isolated .NET assembly context, verifies its SHA against the component record,
and invokes only the pure CPU `ReadQualification` method. The frozen R23 runtime
directory must contain that DLL's managed dependencies. No constructor or device
entry point is invoked.

```sh
dotnet build eng/validation/NinferSmallAttentionAudit/NinferSmallAttentionAudit.csproj -c Release
dotnet eng/validation/NinferSmallAttentionAudit/bin/Release/net10.0/NinferSmallAttentionAudit.dll \
  --packet /ABS/artifacts/ninfer-small-attention-r23 \
  --expect-component-sha256 48029a1e054a46e73cd78570c112dc7e7f11de03133220adf39cfda27a90b4e2 \
  /ABS/artifacts/FRESH-r23-audit.json \
  --frozen-probe /ABS/FROZEN-R23/NinferSmallAttentionProbe.dll
```

The optional local-producer mode (omit `--frozen-probe`) is diagnostic only
unless that DLL has the actual R23 image SHA. Strict replay can fail across
operating systems when FP64 transcendental results differ in their last bits.
That remains a failed reconstruction check: the output separately reports every
exact fixture mismatch and the raw numerical/timing results. It never changes
the fixed envelope, rewrites a failed score or treats local nonexecution as a
new GPU or end-to-end pass.
