# Building TinyTensorCompiler

From this directory, use the Makefile:

```sh
make       # Build, recompiling only what changed.
make run   # Build if needed, then run.
make clean # Remove the executable and generated build files.
```

The Makefile tracks header dependencies automatically using generated `.d` files.

To build manually:

```sh
clang++ -std=c++17 -Wall -Wextra main.cpp Tensor.cpp Operation.cpp Ops.cpp Optimizer.cpp -o main
./main
```

Compile all five source files together. Headers declare interfaces; source files define their implementations.

- `Tensor.h` / `Tensor.cpp`: numerical storage, shape validation, and graph links.
- `Operation.h` / `Operation.cpp`: base operation and connection updates.
- `Ops.h` / `Ops.cpp`: Add, MatMul, ReLU, and fused operations.
- `Optimizer.h` / `Optimizer.cpp`: traversal, fusion patterns, builders, and rewriting.
- `main.cpp`: construct, execute, optimize, and compare the example.

Headers use `#pragma once` to prevent repeated definitions within a compilation unit. Definitions such as `Tensor::print()` in a source file implement methods declared in the corresponding header.
