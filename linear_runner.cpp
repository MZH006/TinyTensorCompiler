#include "Graph.h"
#include "Ops.h"
#include "Optimizer.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Protocol: M K N, then row-major x[M*K], weights[K*N], bias[M*N].
// A bounded validation interface, not a general model importer.
std::vector<float> readValues(std::size_t count) {
    std::vector<float> values(count);
    for (float& value : values) {
        if (!(std::cin >> value) || !std::isfinite(value)) {
            throw std::invalid_argument("Expected finite tensor values");
        }
    }
    return values;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3 ||
            (std::string(argv[1]) != "--fused" && std::string(argv[1]) != "--unfused") ||
            (argc == 3 && std::string(argv[2]) != "--reverse-add")) {
            throw std::invalid_argument(
                "Usage: linear-runner --fused|--unfused [--reverse-add]");
        }
        long long m, k, n;
        if (!(std::cin >> m >> k >> n) || m < 1 || k < 1 || n < 1 ||
            m > 256 || k > 256 || n > 256) {
            throw std::invalid_argument("Dimensions must be integers in [1, 256]");
        }
        const auto M = static_cast<std::size_t>(m);
        const auto K = static_cast<std::size_t>(k);
        const auto N = static_cast<std::size_t>(n);
        Tensor x("x", readValues(M * K), {M, K});
        Tensor weights("weights", readValues(K * N), {K, N});
        Tensor bias("bias", readValues(M * N), {M, N});
        std::string extra;
        if (std::cin >> extra) {
            throw std::invalid_argument("Unexpected trailing input");
        }

        Graph graph;
        auto* mm = graph.addOperation(std::make_unique<MatMul>(&x, &weights));
        auto* add = graph.addOperation(argc == 3
            ? std::make_unique<Add>(&bias, mm->getOutput())
            : std::make_unique<Add>(mm->getOutput(), &bias));
        auto* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
        graph.setOutput(relu->getOutput());
        if (std::string(argv[1]) == "--fused") {
            optimize(graph, {{"linear_relu", {"MatMul", "Add", "ReLU"},
                              buildLinearReLUPattern}});
        }
        graph.execute();
        const auto& values = graph.getOutput()->getData();
        for (float value : values) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite output");
        }
        // max_digits10 preserves a float's value when Python parses the text.
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"shape\":[" << M << ',' << N
                  << "],\"operation_count\":" << graph.getExecutionOrder().size()
                  << ",\"data\":[";
        for (std::size_t i = 0; i < values.size(); i++) {
            if (i) std::cout << ',';
            std::cout << values[i];
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
