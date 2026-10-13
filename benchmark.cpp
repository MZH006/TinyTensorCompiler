#include "Graph.h"
#include "Ops.h"
#include "Optimizer.h"
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <random>

double timeBatchMicroseconds(Graph& graph) {
        constexpr int repetitions = 50;

        const auto start = std::chrono::steady_clock::now();
        for (int run = 0; run < repetitions; run++) {
            graph.execute();
        }
        const auto end = std::chrono::steady_clock::now();
        double microseconds = std::chrono::duration<double, std::micro>(end - start).count()/ repetitions;
        return microseconds;

}

void buildLinearGraph(Graph& graph, Tensor* x, Tensor* weights, Tensor* bias) {
    Operation* matmul = graph.addOperation(std::make_unique<MatMul>(x, weights));
    Operation* add = graph.addOperation(std::make_unique<Add>(matmul->getOutput(), bias));
    Operation* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    graph.setOutput(relu->getOutput());
}

bool outputsMatch(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
            std::abs(a[i] - b[i]) > 1e-5f + 1e-4f * std::abs(a[i])) {
            return false;
        }
    }
    return true;
}

void benchmarkShape(std::size_t M, std::size_t K, std::size_t N) {
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<float> x_data(M * K);
    std::vector<float> weights_data(K * N);
    std::vector<float> bias_data(M * N);

    for (float& value : x_data) {
        value = distribution(generator);
    }

    for (float& value : weights_data) {
        value = distribution(generator);
    }

    for (float& value : bias_data) {
        value = distribution(generator);
    }


    Tensor x("x", x_data, {M, K});
    Tensor weights("weights", weights_data, {K, N});
    Tensor bias("bias", bias_data, {M, N});
    Graph unfused;
    Graph fused;
    buildLinearGraph(unfused, &x, &weights, &bias);
    buildLinearGraph(fused, &x, &weights, &bias);

    const std::vector<FusionPattern> patterns{
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern}
    };

    // Optimize before executing: detached intermediates never receive data.
    optimize(fused, patterns);
    if (unfused.getExecutionOrder().size() != 3 ||
        fused.getExecutionOrder().size() != 1 ||
        fused.getOutput()->getProducer()->getName() != "FusedLinearReLU") {
        throw std::runtime_error("Benchmark graphs have unexpected structure");
    }
    unfused.execute();
    fused.execute();

    if (!outputsMatch(unfused.getOutput()->getData(), fused.getOutput()->getData())) {
        throw std::runtime_error("Benchmark outputs do not match");
    }


    std::vector<double> unfused_timings;
    std::vector<double> fused_timings;

    for (int sample = 0; sample < 21; sample++) {
        if (sample % 2 == 0) {
            // Measure unfused, then fused. Append each result.
            double unfused_time = timeBatchMicroseconds(unfused);
            double fused_time = timeBatchMicroseconds(fused);
            unfused_timings.push_back(unfused_time);
            fused_timings.push_back(fused_time);
        } else {
            double fused_time = timeBatchMicroseconds(fused);
            double unfused_time = timeBatchMicroseconds(unfused);
            unfused_timings.push_back(unfused_time);
            fused_timings.push_back(fused_time);
        }
    }
    std::sort(unfused_timings.begin(), unfused_timings.end());
    std::sort(fused_timings.begin(), fused_timings.end());
    double unfused_us = unfused_timings[unfused_timings.size()/2];
    double fused_us = fused_timings[fused_timings.size()/2];

    std::cout << "X shape: " << M << " " << K << "\n"; 
    std::cout << "Weights Shape: " << K << " " << N << "\n";
    std::cout << "bias shape: " << M << " " << N << "\n";
    std::cout << std::fixed << std::setprecision(3)
              << "Outputs match: yes\n"
              << "Unfused median: " << unfused_us << " us\n"
              << "Fused median:   " << fused_us << " us\n";
    if (fused_us > 0.0) {
        std::cout << "Unfused / fused: " << unfused_us / fused_us << "x\n";
    }
    // MatMul and Add each materialize an M x N float tensor in the unfused graph.
    // This counts tensor payloads, not temporary allocations or process memory.
    const std::size_t avoided_bytes = 2 * M * N * sizeof(float);
    std::cout << "Intermediate tensor bytes avoided: " << avoided_bytes << "\n";
    std::cout << "21 samples per graph, 50 executions per batch; construction and optimization excluded.\n"
              << "Steady-state execution uses cached order and reusable output buffers; includes numerical loops.\n"
              << "Single Shape: timings are exploratory, not a general speedup claim.\n";
}

int main() {
    benchmarkShape(8, 16, 8);
    benchmarkShape(32, 64, 32);
    benchmarkShape(64, 128, 64);
}