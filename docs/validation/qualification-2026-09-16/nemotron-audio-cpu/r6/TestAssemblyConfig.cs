// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp

using Xunit;

// Tests exercise process-global CUDA contexts, native graph caches, and TS_*
// environment overrides. Cross-class parallelism makes those shared resources
// race (CUDA error 700 or a request observing another test's override), so run
// test collections serially for deterministic correctness.
[assembly: CollectionBehavior(DisableTestParallelization = true)]

// Trait taxonomy, selectable with dotnet test --filter:
//   Category=Bench    timing/throughput assertions — a failure means the perf
//                     regressed or the box was busy, never that code is wrong
//   Requires=Cuda     needs a CUDA device
//   Requires=Mlx      needs the MLX native backend (macOS)
//   Requires=Models   needs real GGUF weights (TS_TEST_MODEL_DIR and friends)
// Requires traits come from the gated attributes in GatedFacts.cs
// ([CudaFact], [MlxFact], [ModelFact], ...), which also skip visibly when the
// prerequisite is missing. Untagged tests are self-contained correctness
// tests that run anywhere.
// Common lanes:
//   --filter "Category!=Bench&Requires!=Models&Requires!=Cuda&Requires!=Mlx"   fast inner loop
//   --filter "Category!=Bench"                                                 full correctness
//   --filter "Category=Bench"                                                  perf, on a quiet box
