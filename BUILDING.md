# Building TinyTensorCompiler

From this directory, use the Makefile:

```sh
make       # Build, recompiling only what changed.
make run   # Build if needed, then run.
make test  # Build and run correctness tests.
make test-ubsan # Run tests with UndefinedBehaviorSanitizer.
make test-asan  # Run tests with AddressSanitizer (Linux recommended).
make benchmark # Run the optimized fused/unfused benchmark.
make clean # Remove the executable and generated build files.
```

The Makefile tracks header dependencies automatically using generated `.d` files.

To build manually:

```sh
clang++ -std=c++17 -Wall -Wextra main.cpp Tensor.cpp Operation.cpp Ops.cpp Optimizer.cpp Graph.cpp -o main
./main
```

Compile all six source files together. Headers declare interfaces; source files define their implementations.

- `Tensor.h` / `Tensor.cpp`: numerical storage, shape validation, and graph links.
- `Operation.h` / `Operation.cpp`: base operation and connection updates.
- `Ops.h` / `Ops.cpp`: Add, MatMul, ReLU, and fused operations.
- `Optimizer.h` / `Optimizer.cpp`: traversal, fusion patterns, builders, and rewriting.
- `Graph.h` / `Graph.cpp`: operation ownership, final output tracking, and execution.
- `main.cpp`: construct, execute, optimize, and compare the example.

Headers use `#pragma once` to prevent repeated definitions within a compilation unit. Definitions such as `Tensor::print()` in a source file implement methods declared in the corresponding header.

Graph owns original and fused operations. Input tensors are caller-owned and must outlive Graph. Detached operations are retained until Graph destruction; reclamation is deferred. Graph currently tracks a single final output.

## Correctness tests

`make test` runs 16 named test groups, including 32 deterministic numerical cases against a double-precision reference. Coverage includes both fusion rules, final-output replacement, branches, repeated inputs, downstream rewiring, invalid shapes, missing data, matcher bounds, repeated optimization, input updates, zero-sized dimensions, and ownership/address stability. Numerical comparisons use absolute and relative tolerances. Assertions are required; the tests reject builds with `NDEBUG`.

`make test-ubsan` builds a separate executable that stops on undefined-behavior reports. AddressSanitizer remains unverified on this machine because its runtime previously failed during initialization.

These tests cover the current valid-DAG, finite-float, single-output API. They do not establish support for cyclic graphs, dimension-product overflow, allocation-failure recovery, arbitrary custom rewrite builders, or accelerator execution.

## Benchmark

Run `make benchmark` to compile every benchmark source with `-O3`, separately from normal build objects. Three matrix cases are measured: `(M,K,N) = (8,16,8), (32,64,32), (64,128,64)`. Input data is generated with seed 42 and shared between distinct fused/unfused graphs. The fused graph is optimized before its first execution. Graph structure and output equivalence are checked before timing.

Each graph receives a warmup. The benchmark then collects 21 samples of 50 executions per graph, alternating which graph is measured first. Each sample is divided by 50; the median of those batch averages is reported in microseconds per execution. Construction and optimization are excluded. Traversal, temporary allocations, output copying, and numerical execution remain included. No printing occurs inside the measurement loop.

Intermediate tensor payload bytes avoided are calculated as `2 * M * N * sizeof(float)`. This represents the two eliminated matrix outputs, not process memory, peak allocation, or measured memory traffic. Detached nodes remain owned; optimizing before their first execution prevents them from accumulating old numerical data. See [README.md](README.md) for an illustrative run and interpretation. Shared CI runners are not used for speed thresholds.

## Linux continuous integration

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch. It installs Clang and Make on Ubuntu 24.04, builds the example, runs correctness tests, runs separate UndefinedBehaviorSanitizer and AddressSanitizer executables, and executes the example. Each failing command fails the job. The job has read-only repository permissions and a ten-minute timeout.

Run `make test-asan` on a system with a working AddressSanitizer runtime. CI enables leak detection for this target. Local macOS AddressSanitizer initialization previously failed. Check the repository’s Actions tab for the Linux result; its runtime is independent of the local macOS runtime.

Benchmarks are intentionally manual because shared-runner timings are unsuitable for performance thresholds. Workflow changes must be committed and pushed before GitHub can run them.
