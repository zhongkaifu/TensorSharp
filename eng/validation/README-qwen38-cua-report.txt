Offline CUA aggregation (no model/server/network/GPU execution)

  python3 eng/validation/qwen38-cua-report.py 'docs/validation/qwen38-regression/raw/*cua*.jsonl' --output-dir docs/validation/qwen38-regression/cua-figures --data-only

Remove --data-only to render standard matplotlib PNG+SVG. Matplotlib is imported
only in render(); extraction/tests require the Python standard library only.
Use the existing local figure-venv Python if matplotlib is not in the default
environment. The tool never starts a server or contacts any network endpoint.

Outputs:
  data.json             Sources SHA, preserved headers/launch manifests, cases,
                        every request's original gates, token/cache data and SHA,
                        complete/missing/duplicate identities, qualified metrics.
  summary-zh.json       Brief Chinese summary for ROOT to integrate.
  context-figures.json  List suitable for the overall report's context.figures.
  *.png / *.svg         Optional scientific plots with separate fairness facets.

With no inputs, data-only produces NOT_RUN and no fabricated performance values.
Optional --plan FILE is a JSON list of expected slots. For example:

  [{"engine":"ninfer","mode":"plain","label":"chosen-label","cache_profile":"engine-native-global","lanes":2,"scenario":"append"}]

A missing slot becomes NOT_RUN, with every qualified metric null. The list may
also constrain protocol, steps, context_capacity, fixture_sha256 and deployment_
fingerprint. A nonexistent input file is recorded as NOT_RUN; malformed JSON or
invalid records are ERROR. --include-label / --exclude-label are repeatable globs.

Group identity is engine/mode/label/cache_profile/client lanes/scenario. Different
protocols, step counts, context capacities, fixture SHAs and actual deployment
fingerprints are separate strata. Source records and all original headers remain
available. Exact duplicate raw files/segments are aliases, not independent runs;
conflicting snapshots with one run header are ineligible. Multiple partial runs
never fill one another's missing steps, and a failed attempt is not discarded to
keep a better passing attempt of the same configuration.

Default cases require the exact30step x lane identities and30 wave identities;
the independent edit_media protocol requires its declared6..10 steps (the prepared
small probe uses6). IDs must be integer, in range, unique and complete. Run end
and original case summaries are crosschecked. All seven original quality checks
and three measurement checks must PASS, with complete transport, finite positive
timings and actual server usage. Body/common/model-only canonical/wire hashes
and raw transport SHA are checked; errors withhold aggregation, never regrade an
original FAIL as PASS. Untrimmed bodies and replies remain in the SHA-cited raw.

case_summary and run_end are not sufficient by themselves. run_end.complete
only means the selected scenarios ended; append alone is not the full default
three-scenario1/2-lane matrix. selected_only_coverage explicitly lists unselected
scenarios. --plan can expose the complete intended matrix without inventing runs.

Only fully qualified groups show TTFT, request E2E, wave wall total and actual
completion-token totals. Per-request raw usage/cache/counts/hashes remain visible
even when aggregation is withheld. Missing cache data is unknown, not a zero
cache hit. Multiple runs use medians of whole-run case statistics. The30 related
steps are never treated as30 independent experiments. No CI, significance claim
or tool-stream decode tok/s is computed, and labels are alphabetical rather than
ranked. Different output lengths remain visible alongside timings.

default-lineage is a separate diagnostic group. Explicit trusted-single-tenant-
shared and external engine-native-global may share a comparison facet. This does
not change or prove engine cache settings; actual launch provenance is retained.
Prescribed history can fail emitted-prose ownership proof, so these results do
not represent normal generated-history lineage or the private screenshot task.

CPU mock tests:

  python3 -m unittest discover -s eng/validation/tests -p test_qwen38_cua_report.py -v

The frozen CUA producer, benchmark/scoring, production files and ggml are not
modified. Report generation success is not a quality PASS; inspect summary/status
and each group's qualified flag. Missing, failed and incomplete performance is
null and produces no plotted point or invented zero.
CUA JSONL is a separate schema: feed it to this tool, not the frozen engine-bench
scorer. Embed context-figures.json into the overall context.figures separately.
