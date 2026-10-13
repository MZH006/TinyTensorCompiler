# Execution planning and buffer reuse comparison

Local measurement on October 12, 2026, macOS arm64, Apple Clang 15.0.0, `-O3`.

Before: `414db20`. After: `7183d4d`. Both commits were extracted to temporary directories, built with identical flags and the same byte-identical `benchmark.cpp`. The working checkout was not switched. Five whole-program runs alternated before/after order; each program warmed both graphs and reported a median of 21 batches of 50 executions per mode and shape. The table gives the median of those five reported medians, in microseconds per execution. Construction and optimization are excluded. Before includes per-run DFS and temporary result allocation/copying; after reuses the execution plan and output vectors. Inputs stay constant during timing.

| M × K × N | Before unfused µs | After unfused µs | Before / after | Before fused µs | After fused µs | Before / after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8x16x8 | 0.702 | 0.332 | 2.11× | 0.450 | 0.320 | 1.41× |
| 32x64x32 | 26.127 | 26.551 | 0.98× | 27.273 | 25.802 | 1.06× |
| 64x128x64 | 279.248 | 276.493 | 1.01× | 280.208 | 279.198 | 1.00× |

Small-graph overhead improved in this measurement. Larger workloads changed little or inconsistently; the naive MatMul loops dominate. One round was substantially slower for several cases. All rounds, including that round, are retained in the adjacent JSON file; no samples were discarded. These are exploratory local measurements, not statistical confidence intervals or portable speedup guarantees. Printed harness precision is 0.001 µs. The comparison measures caching and buffer reuse together and does not isolate their individual effects. Output equivalence and graph structure were checked by the harness on every run. No allocator instrumentation was used; buffer-address stability and plan-rebuild counts are verified by the C++ tests.

The current fused/unfused ratio is about 1.04×, 1.03×, and 0.99× across these shapes. Fusion adds little timing benefit here once execution overhead is reduced, though it still removes two intermediate tensor payloads.
