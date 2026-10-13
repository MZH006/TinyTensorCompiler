#include "Codegen.h"
#include "Graph.h"
#include "Ops.h"

#include <ostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

void emitCpp(const Graph& graph, std::ostream& out) {
    const auto order = graph.getExecutionOrder();
    if (order.size() != 1) {
        throw std::invalid_argument("Codegen currently requires one FusedLinearReLU");
    }
    const auto* fused = dynamic_cast<const FusedLinearReLU*>(order.front());
    if (fused == nullptr) {
        throw std::invalid_argument("Codegen currently requires one FusedLinearReLU");
    }

    const auto& inputs = fused->getInputs();
    if (inputs.size() != 3) {
        throw std::invalid_argument("FusedLinearReLU must have three inputs");
    }
    const Tensor* x = inputs[0];
    const Tensor* weights = inputs[1];
    const Tensor* bias = inputs[2];

    const auto& xShape = x->getShape();
    const auto& weightsShape = weights->getShape();
    if (xShape.size() != 2 || weightsShape.size() != 2 ||
        xShape[1] != weightsShape[0] ||
        bias->getShape() != std::vector<std::size_t>{xShape[0], weightsShape[1]}) {
        throw std::invalid_argument("Invalid shapes for linear code generation");
    }
    const auto M = xShape[0];
    const auto K = xShape[1];
    const auto N = weightsShape[1];

    if (M == 0 || K == 0 || N == 0 ||
        M > std::numeric_limits<std::size_t>::max() / K ||
        K > std::numeric_limits<std::size_t>::max() / N ||
        M > std::numeric_limits<std::size_t>::max() / N) {
        throw std::invalid_argument("Codegen requires nonempty shapes without size overflow");
    }
    if (x == weights || x == bias || weights->getProducer() || bias->getProducer()) {
        throw std::invalid_argument("Codegen requires separate constant weights and bias");
    }
    // Build the entire source before writing, so validation cannot emit partial code.
    std::ostringstream source;
    source.imbue(std::locale::classic());
    source << std::scientific << std::setprecision(std::numeric_limits<float>::max_digits10);
    source << "// Generated FusedLinearReLU; no compiler runtime dependencies.\n"
           << "#include <cmath>\n#include <cstddef>\n#include <iomanip>\n"
           << "#include <iostream>\n#include <limits>\n#include <string>\n#include <vector>\n\n"
           << "int main() {\n"
           << "    constexpr std::size_t M = " << M << ", K = " << K << ", N = " << N << ";\n";
    auto emitConstant = [&](const char* name, const Tensor* tensor, std::size_t count) {
        const auto& data = tensor->getData();
        if (data.size() != count) {
            throw std::invalid_argument("Codegen constant size mismatch");
        }
        source << "    const std::vector<float> " << name << "{";
        for (std::size_t i = 0; i < data.size(); ++i) {
            if (!std::isfinite(data[i])) {
                throw std::invalid_argument("Codegen constants must be finite");
            }
            if (i != 0) source << ", ";
            source << data[i] << 'f';
        }
        source << "};\n";
    };
    emitConstant("weights", weights, K * N);
    emitConstant("bias", bias, M * N);
    source << R"CPP(    std::vector<float> x(M * K);
    for (float& value : x) {
        if (!(std::cin >> value) || !std::isfinite(value)) {
            std::cerr << "Expected M * K finite input values\n";
            return 1;
        }
    }
    std::string extra;
    if (std::cin >> extra) {
        std::cerr << "Unexpected extra input value\n";
        return 1;
    }
    std::vector<float> output(M * N);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                sum += x[i * K + k] * weights[k * N + j];
            }
            sum += bias[i * N + j];
            output[i * N + j] = sum < 0.0f ? 0.0f : sum;
            if (!std::isfinite(output[i * N + j])) {
                std::cerr << "Nonfinite execution result\n";
                return 1;
            }
        }
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float value : output) std::cout << value << ' ';
    std::cout << '\n';
}
)CPP";
    out << source.str();
    if (!out) {
        throw std::runtime_error("Failed to write generated source");
    }
}
