# External dependencies

- Keep ggml upstream sources unchanged. Never add or apply ggml patches, rewrite
  fetched ggml files, or make TensorSharp builds depend on a modified ggml tree.
- Implement behavior that ggml does not provide in TensorSharp-owned code,
  including native kernels and backend integration when necessary.
- Validate native changes against an unchanged upstream checkout. Record the
  dependency revision, actual test coverage, and benchmark limitations; do not
  count skipped or unavailable model/device scenarios as passing validation.
