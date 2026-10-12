# Building TinyTensorCompiler

From this directory, use the Makefile:

```sh
make       # Build, recompiling only what changed.
make run   # Build if needed, then run.
make test  # Build and run correctness tests.
make test-ubsan # Run tests with UndefinedBehaviorSanitizer.
make test-asan  # Run tests with AddressSanitizer (Linux recommended).
make benchmark # Run the optimized fused/unfused benchmark.
make test-pytorch # Optional: requires Python with PyTorch installed.
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

## Optional PyTorch validation

Use Python 3.12 with `python3 -m pip install -r requirements-validation.txt`, then run `make test-pytorch`. If PyTorch is already installed, the Make target uses that environment without installing anything. To select another interpreter, use `make test-pytorch PYTHON=/path/to/python`.

The target builds `linear-runner` from `linear_runner.cpp` and the shared compiler objects, then runs `scripts/validate_pytorch.py`. `make clean` removes its executable/object/dependency files. The validation dependency is optional and does not affect `make`, the example, or C++ correctness/sanitizer tests. The [README](README.md#pytorch-numerical-validation) documents the text-input/JSON-output protocol and comparison tolerances.

The `pytorch-validation` CI job runs independently of the existing build/sanitizer job. It selects Python 3.12, caches pip downloads using `requirements-validation.txt`, installs pinned PyTorch 2.9.1 from the official CPU wheel index, and runs `make test-pytorch PYTHON=python`. This job's first successful GitHub execution must be verified after committing and pushing the workflow changes.

## Text graph importer

Build with `make graph-import`, then run `./graph-import example.graph`. The file defines rank-two input tensors, MatMul/Add/ReLU operations, and one output. Tensor declarations use `tensor name rows columns data...`; operations use `matmul out lhs rhs`, `add out lhs rhs`, and `relu out input`, followed by `output name`. Names must be defined before use and cannot be redefined. Blank lines are allowed; comments are not supported.

The importer owns input tensors, validates syntax and shapes, and reports instruction errors with line numbers and exit status 1. It executes before and after fusion, prints operation counts and results, and compares finite values with `atol=1e-5`, `rtol=1e-4`. The example fuses three operations to one and produces `8 0`. `make test-import` checks two valid graphs and 15 invalid files using only Python's standard library. Linux CI includes these checks after the workflow is pushed. `make clean` removes importer build artifacts.
