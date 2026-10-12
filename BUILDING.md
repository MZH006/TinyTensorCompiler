# Building TinyTensorCompiler

From this directory, use the Makefile:

```sh
make       # Build, recompiling only what changed.
make run   # Build if needed, then run.
make test  # Build and run correctness tests.
make test-ubsan # Run tests with UndefinedBehaviorSanitizer.
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
